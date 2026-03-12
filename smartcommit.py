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
    You are a strictly formatted Git commit generator. Analyze the following code diff and generate a single commit message using the Conventional Commits standard.

    Allowed prefixes: feat:, fix:, docs:, style:, refactor:, chore:

    Rules:
    - ONLY output the commit message. No conversational text.
    - Do not wrap the output in quotes.
    - Incorporate any additional context provided by the developer.

    Examples:
    Diff: + function calculateTotal(a, b) {{ return a + b; }}
    Output: feat: add calculateTotal function, which adds a and b and returns the total.

    Actual Diff to analyze:
    {diff}
    {context_instruction}
    
    Analyze the following code diff and generate a single commit message using the Conventional Commits standard.

    Allowed prefixes: feat:, fix:, docs:, style:, refactor:, chore:

    Output:
    """

    print("Analyzing diff...")
    response = await session.respond(prompt)

    commit_msg = response.strip().strip('"').strip("'")

    print(f"\nSuggested commit: \033[92m{commit_msg}\033[0m")
    user_input = input("Accept this commit message? (y/n): ")

    if user_input.lower() == 'y':
        subprocess.run(['git', 'commit', '-m', commit_msg])
        print("✅ Committed successfully!")
    else:
        print("Commit aborted.")


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