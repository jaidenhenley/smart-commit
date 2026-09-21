import os
import re
import sys
import subprocess
import asyncio
import argparse
import textwrap

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MAX_DIFF_CHARS = 16000
MAX_OUTPUT_TOKENS = 600
GENERATION_TIMEOUT_SECONDS = 120
MAX_FEEDBACK_CHARS = 500
MAX_FEEDBACK_ITEMS = 5
PROTECTED_BRANCHES = {"main", "master", "develop", "production"}
LARGE_COMMIT_FILE_THRESHOLD = 20
LARGE_COMMIT_LINE_THRESHOLD = 500
SENSITIVE_FILES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "credentials.json", "secrets.json"}
SENSITIVE_FILE_PATTERNS = [re.compile(r'\.pem$'), re.compile(r'\.key$'), re.compile(r'\.p12$')]
SECRET_PATTERNS = [
    re.compile(r'(?i)(api_key|secret|password|token|private_key)\s*=\s*["\']?\S+'),
    re.compile(r'(?i)(AKIA|sk-|ghp_|xox[baprs]-)\S{10,}'),
]
PROGRESS_BAR_WIDTH = 30

COMMIT_TYPES = ("feat", "fix", "refactor", "perf", "docs", "test", "build", "ci", "style", "chore")
SUBJECT_RE = re.compile(r'^(' + '|'.join(COMMIT_TYPES) + r')(\([a-z0-9._/-]+\))?: (.+)$')
MAX_SUBJECT_CHARS = 72
MAX_BODY_BULLETS = 8

LOCKFILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock",
    "Cargo.lock", "poetry.lock", "Pipfile.lock", "uv.lock", "composer.lock",
    "Gemfile.lock", "go.sum", "flake.lock", "mix.lock", "packages.lock.json",
}
GENERATED_PATH_PATTERNS = [re.compile(p) for p in (
    r'\.min\.(js|css)$',
    r'\.(map|snap)$',
    r'(^|/)(dist|build|out|node_modules|vendor|__generated__|__snapshots__)/',
    r'_pb2(_grpc)?\.py$',
    r'\.pb\.go$',
    r'\.generated\.\w+$',
)]

SPECIAL_TOKENS = {"<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<|eot_id|>"}


def run_git_command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip()


class ProgressBar:
    def __init__(self, total, label="Progress"):
        self.total = max(total, 1)
        self.current = 0
        self.label = label
        self._render()

    def advance(self, step=1, label=None):
        self.current = min(self.total, self.current + step)
        if label:
            self.label = label
        self._render()

    def finish(self, label=None):
        self.current = self.total
        if label:
            self.label = label
        self._render()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _render(self):
        filled = int(PROGRESS_BAR_WIDTH * self.current / self.total)
        bar = '#' * filled + '-' * (PROGRESS_BAR_WIDTH - filled)
        percent = int(100 * self.current / self.total)
        sys.stdout.write(f"\r{self.label} [{bar}] {percent:3d}%")
        sys.stdout.flush()


def normalize_feedback(feedback):
    compact_feedback = " ".join(feedback.split())
    return compact_feedback[:MAX_FEEDBACK_CHARS]


# --- Diff preprocessing -----------------------------------------------------

def split_diff_by_file(raw_diff):
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


def extract_path(file_diff):
    first_line = file_diff.split('\n', 1)[0]
    match = re.match(r'diff --git "?a/(.*?)"? "?b/(.*?)"?$', first_line)
    return match.group(2) if match else None


def split_hunks(file_diff):
    header = []
    hunks = []
    current = None
    for line in file_diff.split('\n'):
        if line.startswith('@@'):
            if current is not None:
                hunks.append(current)
            current = [line]
        elif current is None:
            header.append(line)
        else:
            current.append(line)
    if current is not None:
        hunks.append(current)
    return '\n'.join(header), ['\n'.join(h) for h in hunks]


def is_whitespace_only_hunk(hunk):
    added = ''.join(''.join(l[1:].split()) for l in hunk.split('\n') if l.startswith('+') and not l.startswith('+++'))
    removed = ''.join(''.join(l[1:].split()) for l in hunk.split('\n') if l.startswith('-') and not l.startswith('---'))
    return added == removed


def classify_filtered(path, file_diff):
    basename = path.split('/')[-1]
    if basename in LOCKFILE_NAMES:
        return "lockfile"
    if any(p.search(path) for p in GENERATED_PATH_PATTERNS):
        return "generated file"
    # Binary markers appear only as unprefixed metadata lines; content lines in
    # hunks always start with +/-/space, so skip those to avoid false positives
    # on diffs whose content mentions these strings.
    for line in file_diff.split('\n'):
        if line.startswith(('+', '-', ' ', '\\', '@@')):
            continue
        if line == 'GIT binary patch' or (line.startswith('Binary files ') and line.endswith(' differ')):
            return "binary file"
    return None


def preprocess_diff(raw_diff):
    """Split the raw diff per file, drop lockfiles/generated/binary content and
    whitespace-only hunks. Returns (kept, skipped): kept is [(path, diff_text)],
    skipped is [(path, reason)] — skipped files are named but never sent."""
    kept = []
    skipped = []
    for file_diff in split_diff_by_file(raw_diff):
        path = extract_path(file_diff) or "(unknown file)"
        reason = classify_filtered(path, file_diff)
        if reason:
            skipped.append((path, reason))
            continue
        header, hunks = split_hunks(file_diff)
        meaningful = [h for h in hunks if not is_whitespace_only_hunk(h)]
        if hunks and not meaningful:
            skipped.append((path, "whitespace-only changes"))
            continue
        if len(meaningful) < len(hunks):
            file_diff = '\n'.join([header] + meaningful)
        kept.append((path, file_diff))
    return kept, skipped


def split_file_into_pieces(path, file_diff, max_chars):
    """Split one file's diff at hunk boundaries into pieces <= max_chars.
    A single oversized hunk is split at line boundaries — content is never
    dropped, only divided."""
    if len(file_diff) <= max_chars:
        return [file_diff]

    header, hunks = split_hunks(file_diff)
    units = []
    for hunk in hunks:
        if len(hunk) <= max_chars:
            units.append(hunk)
        else:
            lines = hunk.split('\n')
            part = []
            size = 0
            for line in lines:
                if size + len(line) + 1 > max_chars and part:
                    units.append('\n'.join(part))
                    part = [line]
                    size = len(line) + 1
                else:
                    part.append(line)
                    size += len(line) + 1
            if part:
                units.append('\n'.join(part))

    pieces = []
    current = [header]
    size = len(header)
    for unit in units:
        if size + len(unit) + 1 > max_chars and len(current) > 1:
            pieces.append('\n'.join(current))
            current = [header, unit]
            size = len(header) + len(unit) + 1
        else:
            current.append(unit)
            size += len(unit) + 1
    if len(current) > 1:
        pieces.append('\n'.join(current))
    return pieces


# --- Safety checks ----------------------------------------------------------

def get_current_branch():
    return run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])


def check_protected_branch(branch):
    if branch in PROTECTED_BRANCHES:
        print(f"\033[33mWarning: You are committing directly to '{branch}'.\033[0m")


def check_sensitive_files(all_paths):
    warned = False
    for path in all_paths:
        basename = path.split('/')[-1]
        if basename in SENSITIVE_FILES or any(p.search(basename) for p in SENSITIVE_FILE_PATTERNS):
            print(f"\033[31mWarning: Sensitive file detected in commit: {path}\033[0m")
            warned = True
    return warned


def check_secret_patterns(raw_diff):
    added_lines = [l[1:] for l in raw_diff.splitlines() if l.startswith('+') and not l.startswith('+++')]
    hits = []
    for line in added_lines:
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append(line[:120])
                break
    if hits:
        print(f"\033[31mWarning: Possible secrets or credentials detected in staged changes ({len(hits)} line(s)).\033[0m")
        print("\033[31mReview carefully before committing.\033[0m")
    return bool(hits)


def check_large_commit(raw_diff, num_files):
    added = sum(1 for l in raw_diff.splitlines() if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in raw_diff.splitlines() if l.startswith('-') and not l.startswith('---'))
    total_lines = added + removed

    if num_files >= LARGE_COMMIT_FILE_THRESHOLD:
        print(f"\033[33mWarning: Staging {num_files} files — did you mean to commit everything?\033[0m")
    if total_lines >= LARGE_COMMIT_LINE_THRESHOLD:
        print(f"\033[33mWarning: Large commit ({total_lines} lines changed). Consider splitting into smaller commits.\033[0m")


def warn_unstaged_changes():
    unstaged = [line for line in run_git_command(['git', 'diff', '--name-only']).splitlines() if line]
    untracked = [
        line
        for line in run_git_command(['git', 'ls-files', '--others', '--exclude-standard']).splitlines()
        if line
    ]
    if unstaged:
        print(f"\033[33mWarning: {len(unstaged)} file(s) have unstaged changes not included in this commit.\033[0m")
    if untracked:
        print(f"\033[33mWarning: {len(untracked)} untracked file(s) not included in this commit.\033[0m")


# --- Prompts ----------------------------------------------------------------

FORMAT_SECTION = textwrap.dedent(
    f"""
    Format (Conventional Commits):
    Line 1: <type>: <short summary in imperative mood, lowercase, max {MAX_SUBJECT_CHARS} chars, no trailing period>
    Line 2: blank
    Lines 3+: bullet list of the most important changes, each starting with "- ", at most {MAX_BODY_BULLETS} bullets total

    Allowed types: {', '.join(COMMIT_TYPES)}
    An optional lowercase scope is allowed, e.g. "feat(parser): ...", only when obvious.

    Type guide:
    - feat: new functionality or capability
    - fix: a bug fix or error correction
    - refactor: restructuring with no behavior change
    - perf: performance improvement
    - docs / test / build / ci / style: changes limited to those areas
    - chore: maintenance, config, dependency bumps, minor updates
    """
).strip()


def branch_section(branch):
    if not branch or branch in PROTECTED_BRANCHES or branch == 'HEAD':
        return ""
    return f"\nCurrent branch: {branch} — use it as a hint for the intent of this change.\n"


def skipped_section(skipped_files):
    if not skipped_files:
        return ""
    names = "\n".join(f"- {path} ({reason})" for path, reason in skipped_files)
    return (
        "\nAlso changed, but content excluded from analysis (mention only if relevant, never guess their contents):\n"
        f"{names}\n"
    )


def build_file_summary_prompt(path, diff_text, part_index, total_parts):
    part_note = f" (part {part_index + 1} of {total_parts})" if total_parts > 1 else ""
    return textwrap.dedent(
        f"""
        You are analyzing the staged git diff for the file `{path}`{part_note}.
        Describe what was changed in plain English as a bullet list.

        Rules:
        - ONLY output bullet points. No titles, prefixes, or conversational text.
        - Each bullet starts with "- " and describes the PURPOSE of the change, not the raw code.
        - Good: "- Added secret detection to warn before committing API keys"
        - Bad: "- Added SECRET_PATTERNS = [re.compile(...)]"
        - Group related changes into one bullet instead of listing every variable or line.
        - Only include changes grounded in the diff below. Do not invent anything.
        - Do not wrap output in quotes, backticks, or code fences.

        Diff:
        {diff_text}

        Output:
        """
    ).strip()


def context_and_feedback_sections(developer_context, previous_message, feedback_history):
    context_section = ""
    if developer_context:
        context_section = f"\nAdditional developer context:\n{developer_context}\n"

    previous_message_section = ""
    if previous_message and not feedback_history:
        previous_message_section = (
            "\nPrevious draft to revise. Treat this as raw material, not something you need to preserve:\n"
            f"{previous_message}\n"
        )

    feedback_section = ""
    if feedback_history:
        latest_feedback = feedback_history[-1]
        earlier_feedback = feedback_history[:-1]
        sections = [
            "\nLatest developer feedback. This is the highest-priority instruction and must materially change the draft if possible:\n"
            f"- {latest_feedback}\n"
            "Generate a completely fresh commit message from the changes and feedback. "
            "Do not preserve wording or structure from any earlier draft.\n"
        ]
        if earlier_feedback:
            earlier_lines = "\n".join(f"- {item}" for item in earlier_feedback[-(MAX_FEEDBACK_ITEMS - 1):])
            sections.append(
                "Earlier feedback to keep only if it does not conflict with the latest feedback:\n"
                f"{earlier_lines}\n"
            )
        feedback_section = "".join(sections)

    return context_section, previous_message_section, feedback_section


def build_commit_prompt(change_material, material_label, branch, skipped_files,
                        developer_context=None, previous_message=None, feedback_history=None):
    context_section, previous_message_section, feedback_section = context_and_feedback_sections(
        developer_context, previous_message, feedback_history
    )
    return textwrap.dedent(
        f"""
        You are a strictly formatted Git commit message generator.

        {FORMAT_SECTION}

        Rules:
        - ONLY output the commit message. No conversational text, no explanations.
        - Do not wrap the output in quotes, backticks, or code fences.
        - Do not output template placeholders like {{diff}} or {{context}}.
        - Each bullet is a concise completed action (e.g. "Add X", "Remove Y", "Fix Z").
        - Describe the PURPOSE of changes, never paste raw values, data, UUIDs, object dumps, or code.
        - Deduplicate and merge closely related bullets. Do not copy every input bullet:
          synthesize at most {MAX_BODY_BULLETS} bullets covering the most important changes.
        - Base the message ONLY on the {material_label} below. Do not invent anything.
        {branch_section(branch)}{skipped_section(skipped_files)}{context_section}{previous_message_section}{feedback_section}
        {material_label.capitalize()}:
        {change_material}

        Output:
        """
    ).strip()


def build_retry_prompt(original_prompt, rejected_output, errors):
    error_lines = "\n".join(f"- {e}" for e in errors)
    return (
        f"{original_prompt}\n\n"
        f"Your previous output was rejected:\n{rejected_output}\n\n"
        f"Validation errors:\n{error_lines}\n\n"
        "Output ONLY the corrected commit message, nothing else."
    )


# --- Output validation ------------------------------------------------------

def validate_commit_message(message):
    """Validate the commit message format in code. Returns a list of human-readable
    errors; empty list means valid."""
    errors = []
    if not message or not message.strip():
        return ["Output was empty."]

    if '```' in message:
        errors.append("Output must not contain code fences.")
    if any(token in message for token in SPECIAL_TOKENS):
        errors.append("Output must not contain special tokens.")
    if re.search(r'\{(diff|context|type|scope)\}', message):
        errors.append("Output must not contain template placeholders.")

    lines = message.split('\n')
    subject = lines[0].rstrip()
    match = SUBJECT_RE.match(subject)
    if not match:
        errors.append(
            f"Line 1 must be '<type>: <description>' with type one of: {', '.join(COMMIT_TYPES)} "
            "(lowercase, optional scope in parentheses)."
        )
    else:
        description = match.group(3)
        if description[0].isupper():
            errors.append("The description after the colon must start lowercase.")
    if len(subject) > MAX_SUBJECT_CHARS:
        errors.append(f"Line 1 must be at most {MAX_SUBJECT_CHARS} characters (got {len(subject)}).")
    if subject.endswith('.'):
        errors.append("Line 1 must not end with a period.")

    if len(lines) > 1:
        if lines[1].strip():
            errors.append("Line 2 must be blank, separating the subject from the body.")
        for line in lines[2:]:
            if line.strip() and not line.startswith('- '):
                errors.append('Every body line must be a bullet starting with "- ".')
                break
        num_bullets = sum(1 for line in lines[2:] if line.startswith('- '))
        if num_bullets > MAX_BODY_BULLETS:
            errors.append(
                f"The body has {num_bullets} bullets but at most {MAX_BODY_BULLETS} are allowed — "
                "merge related changes and keep only the most important ones."
            )
        seen = set()
        for line in lines[2:]:
            if line.startswith('- '):
                if line in seen:
                    errors.append("The body contains duplicate bullets.")
                    break
                seen.add(line)

    return errors


def build_fallback_message(kept_files, skipped_files, file_summaries=None):
    """Deterministic commit message built without the model — used when the model's
    output fails validation twice, or when every staged file was filtered out."""
    paths = [path for path, _ in kept_files] or [path for path, _ in skipped_files]
    areas = []
    for path in paths:
        area = path.split('/')[0]
        if area not in areas:
            areas.append(area)
    target = ', '.join(areas[:3])
    if len(areas) > 3:
        target += ' and more'
    subject = f"chore: update {target}"
    if len(subject) > MAX_SUBJECT_CHARS:
        subject = subject[:MAX_SUBJECT_CHARS].rstrip('. ,')

    bullets = []
    summary_by_path = dict(file_summaries or [])
    for path, _ in kept_files[:8]:
        summary = summary_by_path.get(path, "")
        first_bullet = next((l.strip() for l in summary.split('\n') if l.strip().startswith('- ')), None)
        bullets.append(first_bullet if first_bullet else f"- Update {path}")
    if len(kept_files) > 8:
        bullets.append(f"- Update {len(kept_files) - 8} additional file(s)")
    for path, reason in skipped_files[:4]:
        bullets.append(f"- Update {path} ({reason})")

    return subject + "\n\n" + "\n".join(bullets) if bullets else subject


# --- Model ------------------------------------------------------------------

def clean_response(text):
    text = text.strip().strip('"').strip("'")
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    if text.startswith('```'):
        lines = text.split('\n')
        end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
        text = '\n'.join(lines[1:end]).strip()
    return text.strip()


def setup_model(model_name):
    if not MLX_AVAILABLE:
        print("mlx-lm is not installed. Run: pip install mlx-lm")
        sys.exit(1)
    print(f"Loading {model_name}...")
    try:
        model_weights, tokenizer = mlx_load(model_name)
    except Exception as exc:
        print(f"\033[31mError: failed to load model '{model_name}'.\033[0m")
        print(f"\033[31m{exc}\033[0m")
        print("Check the model name, your network connection (first run downloads the weights),")
        print("and free disk space. Set SMARTCOMMIT_MODEL or use --model to pick a different model.")
        sys.exit(1)
    return model_weights, tokenizer


def make_responder(model_weights, tokenizer, timeout_seconds):
    # MLX generation is not thread-safe: concurrent mlx_generate calls abort the
    # process inside Metal. All calls to respond() must be awaited sequentially.
    async def respond(prompt):
        loop = asyncio.get_event_loop()

        def _run():
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                formatted = prompt
            return mlx_generate(model_weights, tokenizer, prompt=formatted,
                                max_tokens=MAX_OUTPUT_TOKENS, verbose=False)

        text = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=timeout_seconds)
        return clean_response(text)

    return respond


# --- Generation flow --------------------------------------------------------

async def summarize_files(kept_files, respond, progress):
    """Summarize each kept file's diff into bullets. Oversized files are split at
    hunk boundaries and summarized in parts. Returns [(path, bullets)]."""
    pieces = []
    for path, file_diff in kept_files:
        parts = split_file_into_pieces(path, file_diff, MAX_DIFF_CHARS)
        for i, part in enumerate(parts):
            pieces.append((path, part, i, len(parts)))

    # Sequential on purpose: mlx_generate is not thread-safe and a single
    # generation already saturates the GPU, so concurrency gains nothing.
    results = []
    for completed, (path, part, part_index, total_parts) in enumerate(pieces, start=1):
        summary = await respond(build_file_summary_prompt(path, part, part_index, total_parts))
        results.append((path, summary))
        progress.advance(label=f"Summarizing changed files ({completed}/{len(pieces)})")

    merged = []
    for path, summary in results:
        if merged and merged[-1][0] == path:
            merged[-1] = (path, merged[-1][1] + "\n" + summary)
        else:
            merged.append((path, summary))
    return merged


async def respond_validated(respond, prompt):
    """Generate, validate in code, retry once with the errors, or return None."""
    message = await respond(prompt)
    errors = validate_commit_message(message)
    if not errors:
        return message
    message = await respond(build_retry_prompt(prompt, message, errors))
    if not validate_commit_message(message):
        return message
    return None


async def build_commit_message(kept_files, skipped_files, branch, developer_context, respond,
                               feedback_history=None, file_summaries=None, previous_message=None,
                               progress_label="Analyzing diff"):
    total_kept_chars = sum(len(diff) for _, diff in kept_files)
    use_summaries = total_kept_chars > MAX_DIFF_CHARS or file_summaries is not None

    if use_summaries:
        if file_summaries is None:
            num_pieces = sum(len(split_file_into_pieces(p, d, MAX_DIFF_CHARS)) for p, d in kept_files)
            progress = ProgressBar(num_pieces + 1, progress_label)
            file_summaries = await summarize_files(kept_files, respond, progress)
        else:
            progress = ProgressBar(1, progress_label)
        progress.advance(0, label="Applying feedback" if feedback_history else "Writing commit message")
        material = "\n".join(f"Changes to {path}:\n{summary}" for path, summary in file_summaries)
        prompt = build_commit_prompt(
            material, "extracted change summaries", branch, skipped_files,
            developer_context=developer_context, previous_message=previous_message,
            feedback_history=feedback_history,
        )
    else:
        progress = ProgressBar(1, progress_label)
        material = "\n".join(diff for _, diff in kept_files)
        prompt = build_commit_prompt(
            material, "staged diff", branch, skipped_files,
            developer_context=developer_context, previous_message=previous_message,
            feedback_history=feedback_history,
        )

    commit_msg = await respond_validated(respond, prompt)
    used_fallback = commit_msg is None
    if used_fallback:
        commit_msg = build_fallback_message(kept_files, skipped_files, file_summaries)
    progress.finish("Updated commit draft ready" if feedback_history else "Commit draft ready")
    if used_fallback:
        print("\033[33mModel output failed format validation twice — using a deterministic fallback message.\033[0m")
    return commit_msg, file_summaries


async def generate_commit_message(developer_context=None, respond=None, dry_run=False, provider_label=""):
    raw_diff = run_git_command(['git', 'diff', '--staged', '--no-color', '--unified=3'])

    if not raw_diff:
        print("No staged changes found. Run `git add` first!")
        warn_unstaged_changes()
        return

    warn_unstaged_changes()
    branch = get_current_branch()
    check_protected_branch(branch)

    kept_files, skipped_files = preprocess_diff(raw_diff)
    all_paths = [p for p, _ in kept_files] + [p for p, _ in skipped_files]
    check_large_commit(raw_diff, len(all_paths))
    has_sensitive_files = check_sensitive_files(all_paths)
    has_secrets = check_secret_patterns(raw_diff)

    if has_sensitive_files or has_secrets:
        confirm = input("Sensitive content detected. Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Commit aborted.")
            return

    if skipped_files:
        skipped_names = ', '.join(path for path, _ in skipped_files)
        print(f"\033[2mExcluded from analysis: {skipped_names}\033[0m")

    if not kept_files:
        commit_msg = build_fallback_message(kept_files, skipped_files)
        file_summaries = None
        print("\033[2mAll staged files were filtered (lockfiles/generated/whitespace) — using a deterministic message.\033[0m")
    else:
        try:
            commit_msg, file_summaries = await build_commit_message(
                kept_files, skipped_files, branch, developer_context, respond,
            )
        except asyncio.TimeoutError:
            print(f"\n\033[31mError: model generation timed out after {GENERATION_TIMEOUT_SECONDS}s. "
                  "No commit was made.\033[0m")
            sys.exit(1)

    feedback_history = []

    while True:
        print(f"\nSuggested commit: \033[92m{commit_msg}\033[0m")
        if provider_label:
            print(f"\033[2m{provider_label}\033[0m")
        user_input = input("Accept? (y), give feedback to regenerate, or abort (n): ").strip()

        if user_input.lower() == 'y':
            if dry_run:
                print("\033[36m[dry-run] Would have committed with message above.\033[0m")
            else:
                result = subprocess.run(['git', 'commit', '-m', commit_msg])
                if result.returncode == 0:
                    print("✅ Committed successfully!")
                else:
                    print(f"\033[31mCommit failed (exit {result.returncode}).\033[0m")
            break
        elif user_input.lower() == 'n':
            print("Commit aborted.")
            break
        elif user_input:
            if not kept_files:
                print("\033[33mAll staged content was filtered out — feedback cannot change the message.\033[0m")
                continue
            feedback_history.append(normalize_feedback(user_input))
            feedback_history = feedback_history[-MAX_FEEDBACK_ITEMS:]

            try:
                commit_msg, _ = await build_commit_message(
                    kept_files, skipped_files, branch, developer_context, respond,
                    feedback_history=feedback_history,
                    file_summaries=file_summaries,
                    previous_message=commit_msg,
                    progress_label="Regenerating commit message",
                )
            except asyncio.TimeoutError:
                print(f"\n\033[31mError: model generation timed out after {GENERATION_TIMEOUT_SECONDS}s. "
                      "No commit was made.\033[0m")
                sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Conventional Commit messages with a local MLX model.")
    parser.add_argument(
        '-c', '--context',
        type=str,
        help='Additional context or intent to guide the model (e.g., "race condition on the login screen")'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.environ.get("SMARTCOMMIT_MODEL", DEFAULT_MODEL),
        help=f'MLX model to use (default: $SMARTCOMMIT_MODEL or {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=int(os.environ.get("SMARTCOMMIT_TIMEOUT", GENERATION_TIMEOUT_SECONDS)),
        help=f'Per-generation timeout in seconds (default: {GENERATION_TIMEOUT_SECONDS})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview the commit message without actually committing'
    )
    args = parser.parse_args()

    GENERATION_TIMEOUT_SECONDS = args.timeout
    model_weights, tokenizer = setup_model(args.model)
    respond = make_responder(model_weights, tokenizer, args.timeout)

    asyncio.run(generate_commit_message(
        developer_context=args.context,
        respond=respond,
        dry_run=args.dry_run,
        provider_label=f"Generated by MLX · {args.model}",
    ))
