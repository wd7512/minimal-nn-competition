# Contributing

## Submissions

Submit a model by opening a pull request against the **`submission`** branch.
Do not target `main`.

Benchmark-specific setup, config fields, and local evaluation:

- [Snake submission](how-to/how-to-make-a-snake-game-submission.md)
- [MNIST submission](how-to/how-to-make-an-mnist-submission.md)

A submission PR must contain exactly two files in a new directory:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

It must not change code, dependencies, workflows, the leaderboard, or `docs/`.
The `Submission check` workflow verifies the changed paths and validates
`config.json`; it does not execute the submitted model. `config.json` must
include non-empty `name` and `author` strings, which are shown on the
leaderboard. You are encouraged to include an optional `description` string
with a short paragraph (or less) explaining how you built the network or its
style. It is not shown on the leaderboard.

If the check fails, update the PR rather than opening a new directory. Once it
passes, a maintainer merges the PR into `submission`.

## Maintainer review

1. Confirm the diff contains only `config.json` and `model.pt` for one
   `models/<benchmark>/<name>/` directory.
2. Evaluate the entry locally on the trusted machine:

   ```bash
   uv run mnc evaluate <benchmark> models/<benchmark>/<name>
   uv run mnc collect --replay-dir replays
   uv run mnc render leaderboard.json docs --replays replays
   ```

3. Commit the updated leaderboard to `submission`.
4. Open a `submission` → `main` pull request and merge it.
5. Fast-forward `submission` to `main` so the next submission starts clean:

   ```bash
   git checkout submission
   git merge --ff-only main
   git push origin submission
   ```

If an entry fails, revert it on `submission` and do not promote it to `main`.

## Enforcing the submission check

In the repository settings, protect the `submission` branch and require the
`validate` status check. This blocks merges unless the PR contains exactly one
submission directory with `config.json` and `model.pt`.

## Other changes

Code, documentation, and workflow changes go through a pull request to `dev`
and are promoted to `main` when ready. `main` requires a pull request;
repository admins can bypass branch protection when necessary. Only submissions
need to go through the `submission` branch.
