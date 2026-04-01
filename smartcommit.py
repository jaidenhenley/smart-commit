import os
import sys
import subprocess
import asyncio
import argparse
import textwrap
import apple_fm_sdk as fm

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

ALLOWED_PREFIXES = "[Feature], [Bug], [Clean], [Patch]"
MAX_DIFF_CHARS = 3000
MAX_FEEDBACK_CHARS = 500
MAX_FEEDBACK_ITEMS = 5
GEMINI_MODEL = "gemini-2.0-flash-lite"
GROQ_MODEL = "llama-3.3-70b-versatile"


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


def split_diff_by_file(raw_diff):
    """Split a raw git diff into a list of per-file diff sections."""
    sections = []
    current = []
    for line in raw_diff.split('\n'):
        if line.startswith('diff --git') and current:
            sections.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append('\n'.join(current))
    return sections


def chunk_file_diffs(file_diffs, max_chars):
    """Group per-file diffs into chunks that each fit within max_chars."""
    chunks = []
    current_chunk = []
    current_size = 0

    for diff in file_diffs:
        if len(diff) > max_chars:
            diff = truncate_at_boundary(diff, max_chars)

        diff_size = len(diff)
        if current_size + diff_size > max_chars and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [diff]
            current_size = diff_size
        else:
            current_chunk.append(diff)
            current_size += diff_size

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def build_chunk_summary_prompt(chunk_diff, chunk_index, total_chunks):
    return textwrap.dedent(
        f"""
        You are analyzing part {chunk_index + 1} of {total_chunks} of a staged git diff.
        Extract a concise bullet list of every change visible in this chunk.

        Rules:
        - ONLY output bullet points. No titles, prefixes, or conversational text.
        - Each bullet starts with "- " and describes a completed action (e.g. "Added X", "Removed Y", "Fixed Z").
        - Only include changes grounded in the diff below. Do not invent anything.
        - Be specific: include function names, variable names, or file names where helpful.
        - Do not wrap output in quotes, backticks, or code fences.

        Diff chunk:
        {chunk_diff}

        Output:
        """
    ).strip()


def build_merge_prompt(all_bullets, developer_context=None, previous_message=None, feedback_history=None):
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
            "Generate a completely fresh commit message from the bullets and feedback. "
            "Do not preserve wording or structure from any earlier draft.\n"
        ]

        if earlier_feedback:
            earlier_lines = "\n".join(f"- {item}" for item in earlier_feedback[-(MAX_FEEDBACK_ITEMS - 1):])
            sections.append(
                "Earlier feedback to keep only if it does not conflict with the latest feedback:\n"
                f"{earlier_lines}\n"
            )

        feedback_section = "".join(sections)

    return textwrap.dedent(
        f"""
        You are a strictly formatted Git commit generator.
        The staged diff was processed in chunks. Below are the extracted change bullets from every chunk.
        Synthesize them into a single well-structured commit message.

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
        - Deduplicate and merge closely related bullets.
        - Each bullet is a concise completed action (e.g. "Added X", "Removed Y", "Fixed Z").
        {context_section}{previous_message_section}{feedback_section}
        Extracted changes from all chunks:
        {all_bullets}

        Output:
        """
    ).strip()


def build_prompt(diff_context, developer_context=None, previous_message=None, feedback_history=None):
    previous_message_section = ""
    if previous_message and not feedback_history:
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
            "Generate a completely fresh commit message from the staged changes and feedback. "
            "Do not preserve wording, structure, or bullets from any earlier draft unless the diff independently supports the same conclusion.\n"
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
        - The latest developer feedback has higher priority than any earlier feedback and your default wording preferences.
        - When feedback is provided, regenerate the entire message from scratch instead of editing or preserving the previous draft.
        - Each bullet is a concise completed action (e.g. "Added X", "Removed Y", "Fixed Z")
        - Only include bullets for things actually supported by the staged changes.
        {context_section}{previous_message_section}{feedback_section}
        Staged changes:
        {diff_context}

        Output:
        """
    ).strip()


def clean_response(text):
    text = text.strip().strip('"').strip("'")
    if text.startswith('```'):
        lines = text.split('\n')
        end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
        text = '\n'.join(lines[1:end]).strip()
    return text


def make_responder(provider, gemini_model=None, groq_client=None):
    """Return an async callable that sends a prompt to the chosen provider."""
    async def respond(prompt):
        if provider == "gemini":
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: gemini_model.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            )
            return clean_response(response.text)
        elif provider == "groq":
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return clean_response(response.choices[0].message.content)
        else:
            session = fm.LanguageModelSession()
            text = await session.respond(prompt)
            return clean_response(text)

    return respond


async def summarize_chunks(chunks, respond):
    total = len(chunks)
    print(f"Diff is large — processing in {total} chunks...")
    tasks = [
        respond(build_chunk_summary_prompt(chunk, i, total))
        for i, chunk in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)
    for i in range(total):
        print(f"  Chunk {i + 1}/{total} analyzed.")
    return "\n".join(results)


async def generate_commit_message(developer_context=None, respond=None, chunking=True, provider_label=""):
    raw_diff = run_git_command(['git', 'diff', '--staged', '--no-color', '--unified=0'])

    if not raw_diff:
        print("No staged changes found. Run `git add` first!")
        return

    use_chunks = chunking and len(raw_diff) > MAX_DIFF_CHARS
    if use_chunks:
        chunks = chunk_file_diffs(split_diff_by_file(raw_diff), MAX_DIFF_CHARS)

    print("Analyzing diff...")

    if use_chunks:
        combined_bullets = await summarize_chunks(chunks, respond)
        print("Merging chunk summaries...")
        commit_msg = await respond(
            build_merge_prompt(combined_bullets, developer_context=developer_context)
        )
    else:
        commit_msg = await respond(
            build_prompt(raw_diff, developer_context=developer_context)
        )

    feedback_history = []

    while True:
        print(f"\nSuggested commit: \033[92m{commit_msg}\033[0m")
        if provider_label:
            print(f"\033[2m{provider_label}\033[0m")
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

            print("Regenerating...")
            if use_chunks:
                commit_msg = await respond(
                    build_merge_prompt(
                        combined_bullets,
                        developer_context=developer_context,
                        previous_message=commit_msg,
                        feedback_history=feedback_history,
                    )
                )
            else:
                commit_msg = await respond(
                    build_prompt(
                        raw_diff,
                        developer_context=developer_context,
                        feedback_history=feedback_history,
                    )
                )


def setup_apple_provider():
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Apple Intelligence unavailable: {reason}")
        sys.exit(1)
    return make_responder("apple")


def setup_groq_provider(api_key):
    if not GROQ_AVAILABLE:
        print("groq is not installed. Run: pip install groq")
        sys.exit(1)
    if not api_key:
        print("Groq API key required. Use --groq-key or set the GROQ_API_KEY environment variable.")
        sys.exit(1)
    client = Groq(api_key=api_key)
    return make_responder("groq", groq_client=client)


def setup_gemini_provider(api_key):
    if not GEMINI_AVAILABLE:
        print("google-genai is not installed. Run: pip install google-genai")
        sys.exit(1)
    if not api_key:
        print("Gemini API key required. Use --gemini-key or set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    return make_responder("gemini", gemini_model=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate smart Git commits using AI.")
    parser.add_argument(
        '-c', '--context',
        type=str,
        help='Additional context or intent to guide the AI (e.g., "fixes ticket #123")'
    )
    parser.add_argument(
        '--provider',
        choices=["apple", "gemini", "groq"],
        default="groq",
        help='AI provider to use (default: groq)'
    )
    parser.add_argument(
        '--groq-key',
        type=str,
        default=None,
        help='Groq API key (or set GROQ_API_KEY env var)'
    )
    parser.add_argument(
        '--gemini-key',
        type=str,
        default=None,
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    args = parser.parse_args()

    if args.provider == "groq":
        api_key = args.groq_key or os.environ.get("GROQ_API_KEY")
        respond = setup_groq_provider(api_key)
        chunking = False
        provider_label = f"Generated by Groq · {GROQ_MODEL}"
    elif args.provider == "gemini":
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        respond = setup_gemini_provider(api_key)
        chunking = False
        provider_label = f"Generated by Gemini · {GEMINI_MODEL}"
    else:
        respond = setup_apple_provider()
        chunking = True
        provider_label = "Generated by Apple Intelligence"

    asyncio.run(generate_commit_message(developer_context=args.context, respond=respond, chunking=chunking, provider_label=provider_label))
