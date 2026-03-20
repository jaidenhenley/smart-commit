import subprocess
import asyncio
import argparse
import sys
import apple_fm_sdk as fm

DIFF_CHAR_LIMIT = 8000


def get_staged_diff():
    diff = subprocess.run(['git', 'diff', '--staged'], capture_output=True, text=True).stdout.strip()
    stat = subprocess.run(['git', 'diff', '--staged', '--stat'], capture_output=True, text=True).stdout.strip()
    return diff, stat


def trim_diff(diff, stat):
    """If diff is too large for the context window, return a stat summary + truncated diff."""
    if len(diff) <= DIFF_CHAR_LIMIT:
        return diff, False

    truncated = diff[:DIFF_CHAR_LIMIT]
    # Don't cut mid-line
    truncated = truncated[:truncated.rfind('\n')]
    return f"[Diff truncated — showing first {DIFF_CHAR_LIMIT} chars]\n\nFile summary:\n{stat}\n\nPartial diff:\n{truncated}", True


def get_recent_commits(n=5):
    result = subprocess.run(
        ['git', 'log', f'-{n}', '--pretty=format:%s'],
        capture_output=True, text=True
    )
    return result.stdout.strip()


async def generate_commit_message(developer_context=None):
    diff, stat = get_staged_diff()

    if not diff:
        print("No staged changes found. Run `git add` first!")
        return

    # Check model availability
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Apple Intelligence unavailable: {reason}")
        return

    session = fm.LanguageModelSession()

    diff_content, was_truncated = trim_diff(diff, stat)
    if was_truncated:
        print("Note: diff is large — sending a summary to fit the context window.")

    recent_commits = get_recent_commits()
    history_section = ""
    if recent_commits:
        history_section = f"\nRecent commit messages from this repo (match this style):\n{recent_commits}\n"

    context_instruction = ""
    if developer_context:
        context_instruction = f"\nAdditional context from the developer to include/consider:\n\"{developer_context}\"\n"

    prompt = f"""
    You are a strictly formatted Git commit generator. Analyze the following code diff and generate a single commit message.

    Allowed prefixes: [Feature], [Bug], [Clean], [Patch]

    Prefix guide:
    - [Feature]: new functionality or capability added
    - [Bug]: a bug fix or error correction
    - [Clean]: refactoring, formatting, or code cleanup with no behavior change
    - [Patch]: small updates, dependency bumps, config changes, or minor fixes

    Rules:
    - ONLY output the commit message. No conversational text.
    - Do not wrap the output in quotes.
    - Start the message with one of the allowed prefixes.
    - Incorporate any additional context provided by the developer.
    {history_section}
    Examples:
    Diff: + function calculateTotal(a, b) {{ return a + b; }}
    Output: [Feature] add calculateTotal function that adds a and b and returns the total

    Actual Diff to analyze:
    {diff_content}
    {context_instruction}

    Analyze the diff and generate a single commit message starting with one of: [Feature], [Bug], [Clean], [Patch]

    Output:
    """

    print("Analyzing diff...")
    response = await session.respond(prompt)

    commit_msg = response.strip().strip('"').strip("'")

    while True:
        print(f"\nChanged files:\n\033[90m{stat}\033[0m")
        print(f"\nSuggested commit: \033[92m{commit_msg}\033[0m")
        user_input = input("Accept? (y), give feedback to regenerate, or abort (n): ").strip()

        if user_input.lower() == 'y':
            subprocess.run(['git', 'commit', '-m', commit_msg])
            print("Committed successfully!")
            break
        elif user_input.lower() == 'n':
            print("Commit aborted.")
            break
        elif user_input:
            feedback_prompt = f"""
    You are a strictly formatted Git commit generator. You previously suggested a commit message that the developer wants revised.

    Previous message: {commit_msg}
    Developer feedback: "{user_input}"

    Allowed prefixes: [Feature], [Bug], [Clean], [Patch]

    Rules:
    - ONLY output the commit message. No conversational text.
    - Do not wrap the output in quotes.
    - Start the message with one of the allowed prefixes.
    - Apply the developer's feedback to improve the message.

    Actual Diff for reference:
    {diff_content}

    Output:
    """
            print("Regenerating...")
            response = await session.respond(feedback_prompt)
            commit_msg = response.strip().strip('"').strip("'")


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