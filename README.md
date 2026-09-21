# Git SmartCommit CLI

**AI-powered Git commit generation running 100% locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).**

SmartCommit analyzes your staged Git changes and generates a [Conventional Commits](https://www.conventionalcommits.org/) message (`feat:`, `fix:`, `refactor:`, ...) with a bullet-list body — validated in code, not just prompted.

* **Private:** your diff never leaves your machine.
* **Free:** no API keys or subscriptions; runs Qwen2.5-Coder-7B-Instruct (4-bit) on-device.
* **Reliable output:** the message format is enforced by a validator. If the model misbehaves twice, a deterministic fallback message is built from the file summaries — you always get a valid commit message, never garbage.

## How it works

1. Grabs the staged diff and strips lockfiles, generated/binary files, and whitespace-only hunks (excluded files are still named to the model, their content is never sent).
2. If the cleaned diff fits the context budget, it's sent in one pass. Larger diffs are summarized per file (split at hunk boundaries — never truncated mid-diff), then the commit message is synthesized from the summaries.
3. The current branch name is included as intent context.
4. The output is validated against the Conventional Commits format. Invalid → one retry with the errors appended → deterministic fallback.
5. You accept (`y`), abort (`n`), or type feedback to regenerate.

## Prerequisites

* Apple Silicon Mac (M1 or newer; 16GB+ RAM recommended)
* Python 3.10+
* `pip install mlx-lm` (the only dependency)

## Installation

```bash
git clone https://github.com/brazill7/smart-commit.git
cd smart-commit
pip install mlx-lm
```

Optional alias so `sc` works anywhere:

```bash
alias sc='python3 /path/to/smart-commit/smartcommit.py'
```

The model weights (~4.3GB) download automatically on first run.

## Usage

Stage your changes first (`git add`), then:

```bash
sc
```

With extra context for the model:

```bash
sc -c "fixes the race condition on the login screen"
```

Preview without committing:

```bash
sc --dry-run
```

### Configuration

| Option | Env var | Default |
|---|---|---|
| `--model` | `SMARTCOMMIT_MODEL` | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| `--timeout` | `SMARTCOMMIT_TIMEOUT` | `120` seconds |

**Example output:**

```
Suggested commit:
feat: add diff preprocessing and output validation

- Filter lockfiles and generated files before sending diffs to the model
- Summarize large diffs per file instead of truncating
- Validate the commit format in code with a deterministic fallback
Accept? (y), give feedback to regenerate, or abort (n):
```
