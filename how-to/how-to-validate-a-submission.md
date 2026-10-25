# How to validate submissions

Submission PRs target `submission`. The acceptance command processes them in
PR-number order, validates one PR, scores it, updates the leaderboard, and
pushes the result before continuing.

Authenticate once with GitHub CLI:

```bash
gh auth login
```

Process every open submission PR:

```bash
uv run mnnc validate-submissions --all
```

Process one PR:

```bash
uv run mnnc validate-submissions --pr <number>
```

For each PR the command:

1. Checks that the submission workflow passed and that the PR changes only
   `models/<benchmark>/<name>/config.json` and `model.pt`.
2. Reads the benchmark from `config.json` and downloads or reuses only its
   required data cache.
3. Scores the candidate in a temporary worktree.
4. Leaves failing PRs open, comments with the failure, and continues.
5. Merges passing PRs, updates `leaderboard.json` and `docs/`, and pushes
   `submission`.

After processing, it creates and merges the promotion PR from `submission` to
`main`, then synchronizes `submission` with `main`. It deletes only the
short-lived `submit/*` head branch of each submission PR; promotion merges
never delete `submission`, `dev`, or `main`.

The command requires a clean worktree and a working `gh auth login` session.
It uses the locked environment and deterministic benchmark inputs. Do not run
submission code directly; checkpoints are loaded by the evaluator with
`weights_only=True`.
