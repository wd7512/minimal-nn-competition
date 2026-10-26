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

For each open PR, the workflow refreshes the current integration tip immediately
before validating that PR, so a branch based on an older tip is rejected even if
another run advanced `origin/submission` during this run. It checks the PR's
validation status, verifies that the tip is an ancestor of the submitted head,
and allows a short bounded retry when GitHub has not reported any checks yet.
Pending or failed checks are not accepted. It then validates that the diff
changes only `models/<benchmark>/<name>/config.json` and `model.pt`, and scores
the candidate in a temporary worktree.

A passing PR is merged without branch cleanup in the critical path. The merge is
pinned to the validated head commit, and the workflow verifies that the merge
commit's second parent matches it before publishing. The workflow then
synchronizes from `origin/submission`, upserts the leaderboard entry by
`(benchmark, submission)`, records a deterministic replay for Snake, CartPole,
Pendulum, or LunarLander, renders the site, commits and pushes the artifacts,
and reads the remote leaderboard back to verify that both the new entry and the
previously recorded entries remain present. If another workflow run advances
`submission` during publication, the generated score commit is rebuilt from the
newer remote ledger, with a bounded retry. An unrelated or ambiguous branch
divergence fails closed rather than being force-pushed.

If a PR has merged but local scoring, rendering, publishing, or verification
fails, the workflow does not label the submission as rejected or post a
validation-failure comment on the merged PR. It attempts recovery immediately.
If that still fails, fix the reported cause and rerun the command with that PR
number; the merged PR is re-scored and its result/replay are upserted
idempotently. Recovery requires the PR to have been merged with a merge commit
(the workflow's merge strategy); PRs merged with squash or rebase must be
reconciled manually rather than scored from a partial diff. Local
branch/worktree cleanup is never allowed to undo acceptance; remote cleanup is
best-effort and limited to short-lived `submit/*` branches; protected branches
and fork branches are left alone.

Publishing stages only the generated artifacts (`docs/leaderboard.json`,
`docs/index.html`, `docs/submissions`, and replay JSON under `docs/games`), so
unrelated files in `docs/` are never committed. Replay recording and site
rendering are best-effort: a presentation failure is logged and the score is
still published, and a later rerun repairs the missing artifacts. Rendered
output is staged inside `docs/` and swapped into place; if an installation step
fails, the previous site tree is restored before the score-only fallback runs.
A score commit left behind by a failed push is verified and published on the
next run; any other unpublished local commit stops the run with a recovery
error instead of being reported as promoted.

Promotion is an independent, idempotent stage that runs on every invocation,
including when there are no open PRs. It resumes an existing open promotion PR
instead of creating a duplicate, verifies that `origin/main` contains
`origin/submission` and every ledger entry, and fast-forwards `submission` after
a promotion merge. If a previous run merged the promotion but failed to
synchronise `submission`, the next run fast-forwards it to the promoted `main`
tip. The workflow never merges `main` into `submission`, and if the branches
have genuinely diverged it stops with a remediation message instead of
attempting a reset, rebase, or force-push automatically.

The command holds an exclusive lock for the checkout, so two runs cannot mutate
the same working tree. It requires a clean tracked worktree and a working
`gh auth login` session. Use the locked environment and deterministic benchmark
inputs. Do not run submission code directly; checkpoints are loaded by the
evaluator with `weights_only=True`.
