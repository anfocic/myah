---
name: ship
description: Review, lint, type-check, test, commit, and open a PR for the current changes on the feature branch.
disable-model-invocation: true
---

# Ship Pipeline (mia)

Follow these steps in strict order. **Stop immediately if any step fails.**

mia workflow: every change lives on a feature branch, lands via PR, never direct to `main`.

Report each step's outcome in one line before moving on, so fole can follow progress. No long monologues.

## Step 1 — Pre-flight

```bash
git status -sb
git diff --stat HEAD
```

- If the working tree is clean and nothing is staged → say "Nothing to ship" and stop.
- If the current branch is `main` → stop with: "Ship runs from a feature branch. Create one with `git checkout -b feat/<slug>` first."
- If on a detached HEAD → say so and stop.
- If there are untracked files, list them and ask fole which belong in the commit before proceeding.

## Step 2 — Review

Launch the **general-purpose** agent with this prompt:

> Review all uncommitted changes in this mia repo as a senior engineer. Read `git diff HEAD` and the changed files. Flag: security issues (secret leaks, injection, path traversal, cwd/vault-guard escapes), correctness bugs (unhandled errors, resource leaks, race conditions), violations of the architecture and rules in `CLAUDE.md`, and anything committed that shouldn't be (debug prints, leftover TODOs, commented-out code). Verify new tools are registered and gated correctly, and that new behavior has test coverage. Report `Verdict: PASS` or `Verdict: FAIL` on the first line, then numbered blockers with `file:line`, then warnings. Under 300 words. Do not edit files.

- **PASS** → proceed to Step 3.
- **FAIL with blockers** → report the issues to fole and stop.
- **FAIL with only warnings** → report them and ask fole whether to proceed.

## Step 3 — Lint

```bash
ruff check .
```

- Clean → proceed to Step 4.
- Errors → report them and stop. (`ruff check --fix` may resolve some; apply only obvious, safe fixes, then re-run.)

## Step 4 — Type-check

```bash
mypy .
```

- Clean → proceed to Step 5.
- Errors → report them and stop. Do not commit type-broken code.

## Step 5 — Test

```bash
python -m pytest -q
```

- All pass → proceed to Step 6.
- Any failure → report failures and stop. If a test failed because the test itself was wrong (not the code), fix the test per `CLAUDE.md`, then re-run from Step 3.

## Step 6 — Commit

1. Stage changed files by name — **never** `git add .` or `git add -A`. Add untracked files individually only if they belong in the commit.
2. Re-check `git diff --cached` for accidentally staged secrets (`.env*`, `*credentials*`, `*token*`, keys). If anything looks sensitive, stop and ask fole.
3. Write a concise Conventional Commits message (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`). Subject ≤ 72 chars. Body only if the *why* isn't obvious from the diff.
4. **No `Co-Authored-By: Claude`, no "Generated with Claude Code", no tool-name signatures** — per global rules. The commit stands on its own.
5. Create the commit. Pass the message via HEREDOC to preserve formatting.
6. If a pre-commit hook rejects the commit (ruff-format, whitespace, etc.): let it apply its fixes or fix the issue yourself, re-stage by name, and create a **new** commit. Never `--amend`, never `--no-verify`.

## Step 7 — Push + PR (ask first)

Ask fole: **"Push and open PR?"**

If yes:

1. `git push -u origin "$(git branch --show-current)"`.
2. Determine the base branch: `gh api repos/:owner/:repo --jq .default_branch` (fallback `main`).
3. Open the PR:
   ```bash
   gh pr create --base <default> --title "<commit subject>" --body "$(cat <<'EOF'
   ## Summary

   <1-3 bullets describing the change>

   ## Test plan

   - [x] ruff check
   - [x] mypy
   - [x] pytest
   EOF
   )"
   ```
4. Return the PR URL.

If no → stop. The commits stay local on the feature branch.

## Rules

- Never skip review, lint, type-check, or tests.
- Never push to `main`. Always branch → PR.
- Never force-push, never `--amend` after push, never `--no-verify`.
- Never `git add .` / `-A` — always explicit file names.
- Never commit anything matching `*.env*`, `*credentials*`, `*secret*`, `*token*` without explicit fole approval.
- No Claude attribution anywhere in commits or PR bodies.
