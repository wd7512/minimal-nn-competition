# Contributing

## Submissions

Submit a model by opening a pull request against the **`submission`** branch.
Do not target `main`.

A submission PR must contain exactly two files in a new directory:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

It must not change code, dependencies, workflows, the leaderboard, or `docs/`.
The `Submission check` workflow verifies the changed paths and validates
`config.json`; it does not execute the submitted model.

If the check fails, update the PR rather than opening a new directory. Once it
passes, a maintainer merges the PR into `submission`.

## Maintainer review

1. Confirm the diff contains only `config.json` and `model.pt` for one
   `models/<benchmark>/<name>/` directory.
2. Evaluate the entry locally on the trusted machine:

   ```bash
   uv run mnc evaluate <benchmark> models/<benchmark>/<name>
   uv run mnc collect
   uv run mnc render leaderboard.json docs
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

Code, documentation, and workflow changes can be committed directly to `main`.
Only submissions need to go through the `submission` branch.
