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
   ```

3. Rebuild the leaderboard as described in [Leaderboard](#leaderboard), then
   commit it to `submission`.
4. Open a `submission` → `main` pull request and merge it.
5. Fast-forward `submission` to `main` so the next submission starts clean:

   ```bash
   git checkout submission
   git merge --ff-only main
   git push origin submission
   ```

If an entry fails, revert it on `submission` and do not promote it to `main`.

## Leaderboard

The site in `docs/` is deployed by `.github/workflows/pages.yml`. Pages must use
"GitHub Actions" as its source. Rebuild it with:

```bash
uv run mnc download-mnist --output data
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

`mnc collect` evaluates every `models/<benchmark>/<name>/` directory.
`--replay-dir` records one seed-0 game per Snake submission; `--replays`
renders each game under `docs/games/`. The leaderboard includes a
`Normalisation` column for MNIST submissions. Inspect a saved leaderboard
without rewriting it:

```bash
uv run mnc leaderboard leaderboard.json
```

The hidden speed board measures the local Snake environment:

```bash
uv run mnc speed
uv run mnc render-speed speed-leaderboard.json docs/internal
```

The internal token leaderboard scores this repo's own code for syntactic
compactness (keywords, operators, and literals count; identifiers, comments,
whitespace, and docstrings do not) across the `core`, `mnist`, and `snake`
scopes. A scope scores only when all of its mapped tests pass:

```bash
uv run mnc internal
uv run mnc render-internal internal-leaderboard.json docs
```

The hidden page lands at `docs/internal/` alongside the speed board.

## Enforcing the submission check

In the repository settings, protect the `submission` branch and require the
`validate` status check. This blocks merges unless the PR contains exactly one
submission directory with `config.json` and `model.pt`.

## Local development

```bash
uv sync --dev --locked
uv run mnc list
```

`mnc`, `minimal-nn-competition`, and `minimal_nn_competition` are equivalent
entry points; `python -m minimal_nn_competition` also works.

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## Other changes

Code, documentation, and workflow changes go through a pull request to `dev`
and are promoted to `main` when ready. `main` requires a pull request;
repository admins can bypass branch protection when necessary. Only submissions
need to go through the `submission` branch.
