import subprocess
import asyncio
import argparse
import sys
import apple_fm_sdk as fm


async def generate_commit_message(developer_context=None):
    # Grab the staged changes
    diff_process = subprocess.run(['git', 'diff', '--staged'], capture_output=True, text=True)
    diff = diff_process.stdout.strip()

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

    # Dynamically build the context section
    context_instruction = ""
    if developer_context:
        context_instruction = f"\nAdditional context from the developer to include/consider:\n\"{developer_context}\"\n"

    # Inject it into the Prompt
    prompt = f"""
    You are a strictly formatted Git commit generator. Analyze the following code diff and generate a detailed commit message.

    Allowed prefixes: [Feature], [Bug], [Clean], [Patch]

    Prefix guide:
    - [Feature]: new functionality or capability added
    - [Bug]: a bug fix or error correction
    - [Clean]: refactoring, formatting, or code cleanup with no behavior change
    - [Patch]: small updates, dependency bumps, config changes, or minor fixes

    Format:
    Line 1: [Prefix] Short summary title
    Lines 2+: Bullet list of completed changes using "- " prefix

    Rules:
    - ONLY output the commit message. No conversational text.
    - Do not wrap the output in quotes.
    - Start the message with one of the allowed prefixes.
    - Incorporate any additional context provided by the developer.
    - Each bullet is a concise completed action (e.g. "Added X", "Removed Y", "Fixed Z")
    - Only include bullets for things actually in the diff — no filler.

    Example:
    Diff: + function calculateTotal(a, b) {{ return a + b; }}
    Output:
    [Feature] Add checkout total calculation

    - Added calculateTotal(a, b) utility function
    - Integrated calculateTotal into the checkout flow

    Actual Diff to analyze:
    {diff}
    {context_instruction}

    Analyze the diff and generate a detailed commit message starting with one of: [Feature], [Bug], [Clean], [Patch]

    Output:
    """

    print("Analyzing diff...")
    response = await session.respond(prompt)

    commit_msg = response.strip().strip('"').strip("'")

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
            # Treat any other input as feedback — regenerate with it
            feedback_prompt = f"""
    You are a strictly formatted Git commit generator. You previously suggested a commit message that the developer wants revised.

    Previous message:
    {commit_msg}

    Developer feedback: "{user_input}"

    Allowed prefixes: [Feature], [Bug], [Clean], [Patch]

    Format:
    Line 1: [Prefix] Short summary title
    Lines 2+: Bullet list of completed changes using "- " prefix

    Rules:
    - ONLY output the commit message. No conversational text.
    - Do not wrap the output in quotes.
    - Start the message with one of the allowed prefixes.
    - Apply the developer's feedback to improve the message.
    - Each bullet is a concise completed action (e.g. "Added X", "Removed Y", "Fixed Z")
    - Only include bullets for things actually in the diff — no filler.

    Actual Diff for reference:
    {diff}

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