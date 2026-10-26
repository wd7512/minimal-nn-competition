# How to validate submissions

Submission PRs target `submission`. The acceptance command processes them in
PR-number order, validates one PR, scores it, updates the leaderboard and game
replays, and pushes the result before continuing.

Authenticate once with GitHub CLI:

```bash
gh auth login
```

Process every open submission PR:

```bash
uv run mnnc validate-submissions --all
```

Process one open PR, or recover a PR that was already merged but whose scoring
or publication step failed:

```bash
uv run mnnc validate-submissions --pr <number>
```

Submission branches must be based on the latest `origin/submission`, not on
`main` or `dev`:

```bash
git fetch origin
git switch -c submit/<name> origin/submission
```

For each open PR, the workflow checks the PR's validation status, verifies that
the current integration tip is an ancestor of the submitted head, and allows a
short bounded retry when GitHub has not reported any checks yet. Pending or
failed checks are not accepted. It then validates that the diff changes only
`models/<benchmark>/<name>/config.json` and `model.pt`, and scores the candidate
in a temporary worktree.

A passing PR is merged without branch cleanup in the critical path. The
workflow then synchronizes from `origin/submission`, upserts the leaderboard
entry by `(benchmark, submission)`, records a deterministic replay for Snake,
CartPole, Pendulum, or LunarLander, renders the site, commits and pushes the
artifacts, and reads the remote leaderboard back to verify that both the new
entry and the previously recorded entries remain present. If another workflow
run advances `submission` during publication, the generated score commit is
rebuilt from the newer remote ledger, with a bounded retry. An unrelated or
ambiguous branch divergence fails closed rather than being force-pushed.

If a PR has merged but local scoring, rendering, publishing, or verification
fails, the workflow does not label the submission as rejected or post a
validation-failure comment on the merged PR. It attempts recovery immediately.
If that still fails, fix the reported cause and rerun the command with that PR
number; the merged PR is re-scored and its result/replay are upserted
idempotently. Local branch/worktree cleanup is never allowed to undo acceptance;
remote cleanup is best-effort and limited to short-lived `submit/*` branches;
protected branches and fork branches are left alone.

After processing, the workflow creates and merges the promotion PR from
`submission` to `main`, then synchronizes `submission` with `main`. It does not
merge `main` into `submission` during acceptance preflight. If `origin/main` is
not an ancestor of `origin/submission`, or if local and remote `submission`
refs have diverged, it stops with a remediation message instead of attempting
a reset, rebase, or force-push automatically.

The command requires a clean tracked worktree and a working `gh auth login`
session. Use the locked environment and deterministic benchmark inputs. Do not
run submission code directly; checkpoints are loaded by the evaluator with
`weights_only=True`.
