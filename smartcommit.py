import subprocess
import asyncio
import argparse
import textwrap
import apple_fm_sdk as fm


ALLOWED_PREFIXES = "[Feature], [Bug], [Clean], [Patch]"
MAX_DIFF_CHARS = 12000
MAX_FEEDBACK_CHARS = 500
MAX_FEEDBACK_ITEMS = 5


def run_git_command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip()


def truncate_at_boundary(text, max_chars):
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars // 2:
        truncated = truncated[:last_newline]

    return truncated.rstrip()


def normalize_feedback(feedback):
    compact_feedback = " ".join(feedback.split())
    return truncate_at_boundary(compact_feedback, MAX_FEEDBACK_CHARS)


def build_diff_context():
    raw_diff = run_git_command(['git', 'diff', '--staged', '--no-color', '--unified=0'])
    if not raw_diff:
        return ""

    if len(raw_diff) <= MAX_DIFF_CHARS:
        return raw_diff

    diff_stat = run_git_command(['git', 'diff', '--staged', '--stat', '--no-color'])
    file_list = run_git_command(['git', 'diff', '--staged', '--name-only', '--no-color'])
    truncated_diff = truncate_at_boundary(raw_diff, MAX_DIFF_CHARS)

    extras = []
    if file_list:
        extras.append(f"Changed files:\n{file_list}")
    if diff_stat:
        extras.append(f"Diff stats:\n{diff_stat}")

    extra_context = "\n\n".join(extras)
    return (
        f"{truncated_diff}\n\n"
        "[Diff truncated because the staged patch is large. Use the visible diff plus the file list/stats below. "
        "Do not invent changes that are not grounded in this context.]"
        f"{f'{chr(10)}{chr(10)}{extra_context}' if extra_context else ''}"
    )


def build_prompt(diff_context, developer_context=None, previous_message=None, feedback_history=None):
    previous_message_section = ""
    if previous_message:
        previous_message_section = (
            "\nPrevious draft to revise. Treat this as raw material, not something you need to preserve:\n"
            f"{previous_message}\n"
        )

    context_section = ""
    if developer_context:
        context_section = f"\nAdditional developer context:\n{developer_context}\n"

    feedback_section = ""
    if feedback_history:
        latest_feedback = feedback_history[-1]
        earlier_feedback = feedback_history[:-1]

        sections = [
            "\nLatest developer feedback. This is the highest-priority instruction and must materially change the draft if possible:\n"
            f"- {latest_feedback}\n"
        ]

        if earlier_feedback:
            earlier_feedback_lines = "\n".join(f"- {item}" for item in earlier_feedback[-(MAX_FEEDBACK_ITEMS - 1):])
            sections.append(
                "Earlier feedback to keep only if it does not conflict with the latest feedback:\n"
                f"{earlier_feedback_lines}\n"
            )

        feedback_section = "".join(sections)

    return textwrap.dedent(
        f"""
        You are a strictly formatted Git commit generator. Analyze the staged changes and generate a detailed commit message.

        Allowed prefixes: {ALLOWED_PREFIXES}

        Prefix guide:
        - [Feature]: new functionality or capability added
        - [Bug]: a bug fix or error correction
        - [Clean]: refactoring, formatting, or code cleanup with no behavior change
        - [Patch]: small updates, dependency bumps, config changes, or minor fixes

        Format:
        Line 1: [Prefix] Short summary title
        Lines 2+: Bullet list of completed changes using "- " prefix

        Rules:
        - ONLY output the commit message. No conversational text, no explanations.
        - Do not wrap the output in quotes, backticks, or code fences.
        - Do not output any template placeholders like {{diff}} or {{context}}.
        - Start the message with one of the allowed prefixes.
        - Use any additional developer context and feedback if provided.
        - The latest developer feedback has higher priority than the previous draft, earlier feedback, and your default wording preferences.
        - Revise aggressively when feedback asks for a different tone, level of detail, focus, or prefix.
        - Keep parts of the previous draft only when they still fit the latest feedback.
        - Each bullet is a concise completed action (e.g. "Added X", "Removed Y", "Fixed Z")
        - Only include bullets for things actually supported by the staged changes.
        - If the diff is truncated, stay grounded in the visible patch plus the file list/stats.
        {context_section}{previous_message_section}{feedback_section}
        Staged changes:
        {diff_context}

        Output:
        """
    ).strip()


async def generate_response(prompt):
    session = fm.LanguageModelSession()
    response = await session.respond(prompt)
    response = response.strip().strip('"').strip("'")
    # Strip markdown code fences the model sometimes wraps output in
    if response.startswith('```'):
        lines = response.split('\n')
        end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
        response = '\n'.join(lines[1:end]).strip()
    return response


async def generate_commit_message(developer_context=None):
    # Grab the staged changes
    diff = build_diff_context()

    if not diff:
        print("No staged changes found. Run `git add` first!")
        return

    # Check model availability
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Apple Intelligence unavailable: {reason}")
        return

    print("Analyzing diff...")
    prompt = build_prompt(diff, developer_context=developer_context)
    commit_msg = await generate_response(prompt)
    feedback_history = []

    while True:
        print(f"\nSuggested commit: \033[92m{commit_msg}\033[0m")
        user_input = input("Accept? (y), give feedback to regenerate, or abort (n): ").strip()

        if user_input.lower() == 'y':
            subprocess.run(['git', 'commit', '-m', commit_msg])
            print("✅ Committed successfully!")
            break
        elif user_input.lower() == 'n':
            print("Commit aborted.")
            break
        elif user_input:
            feedback_history.append(normalize_feedback(user_input))
            feedback_history = feedback_history[-MAX_FEEDBACK_ITEMS:]

            feedback_prompt = build_prompt(
                diff,
                developer_context=developer_context,
                previous_message=commit_msg,
                feedback_history=feedback_history,
            )
            print("Regenerating...")
            commit_msg = await generate_response(feedback_prompt)


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Generate smart Git commits using Apple Intelligence.")

    # Add a -c / --context flag
    parser.add_argument(
        '-c', '--context',
        type=str,
        help='Additional context or intent to guide the AI (e.g., "fixes ticket #123")'
    )

    # Parse the arguments the user typed
    args = parser.parse_args()

    # Run the main function, passing in the context if it exists
    asyncio.run(generate_commit_message(developer_context=args.context))
