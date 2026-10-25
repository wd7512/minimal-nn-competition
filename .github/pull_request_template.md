## What does this PR do?

<!-- Describe the change. -->

## Submission checklist

Only for PRs targeting `submission`:

- [ ] This PR targets `submission`, not `main`.
- [ ] It adds one `models/<benchmark>/<name>/` directory.
- [ ] That directory contains only `config.json` and `model.pt`.
- [ ] No code, dependency, workflow, leaderboard, or `docs/` changes are included.
- [ ] `config.json` matches the directory benchmark and the submission format.

## Maintainer checklist

- [ ] `uv run dev-tools check` passed; the PR description includes its result summary.
- [ ] The submission check passed.
- [ ] I evaluated the submission locally and updated the leaderboard.
- [ ] I promoted the result to `main` and synced `submission` with `main`.
