# Git SmartCommit CLI

**AI-powered Git commit generation running *100% locally* on your Mac using Apple Intelligence.**

SmartCommit is a lightweight CLI tool that analyzes your staged Git changes and generates concise, professional [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, etc.). 

Because it runs completely on-device using the Apple Foundation Models SDK, it offers massive advantages over cloud-based AI tools:

* **Privacy-First:** Your codebase never leaves your machine. Perfect for enterprise, proprietary, or sensitive code.
* **Zero API Costs:** No OpenAI API keys or GitHub Copilot subscriptions required. It uses the AI already built into your Mac.
* ️**Fast:** Powered natively by Apple Silicon neural engines for instant inference.

---

##  Prerequisites

Before installing, ensure your machine meets the hardware and software requirements for local Apple Intelligence:
* **Hardware:** Apple Silicon Mac (M1 chip or newer).
* **OS:** macOS 15.0 (Sequoia) or newer.
* **Settings:** Apple Intelligence must be enabled on your Mac.

---

## Installation

```bash
# 1. Download the latest release
curl -L -o smartcommit githublink

# 2. Make the file executable
chmod +x smartcommit

# 3. Clear the macOS Gatekeeper quarantine flag (required for unsigned binaries)
xattr -d com.apple.quarantine smartcommit

# 4. Move it to your local bin so it can be run from anywhere
sudo mv smartcommit /usr/local/bin/

### (Optional) Set up a Git Alias
If you want to use this tool natively within Git (e.g., typing `git sc` or `git smart-commit` etc. ), you can add a global alias:

git config --global alias.sc '!smartcommit'
git config --global alias.smart-commit '!smartcommit'
```

---

## Usage

Make sure you have staged your changes (`git add .`) before running the tool.

### Option A: Using the Standalone Command
If you skipped the alias step, you can just call the tool directly in your repository:

`smartcommit`

To provide custom context to the AI (like explaining  *why*  you made a change), use the `-c` flag:

`smartcommit -c "race condition on the login screen"`


### Option B: Using the Git Alias
If you configured the `git smart-commit` alias, you can use it just like a native Git command:

`git smart-commit`

With custom context:

`git smart-commit -c "refactored the login auth flow"`


**Example Output:**
> Analyzing diff...
> Suggested commit: **fix: resolve race condition in login flow**
> Accept this commit message? (y/n): 
