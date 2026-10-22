# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

## Problems

- **Snake** — fill an 8×8 board.
- **MNIST** — classify handwritten digits.

## Rules

- Feed-forward networks: `linear` or `mlp`.
- ReLU or another standard activation.
- A PyTorch `state_dict` checkpoint.

## Submit

Open a pull request against the `submission` branch with exactly two files:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

Do not include code, dependency, workflow, leaderboard, or `docs/` changes.

Benchmark guides:

- [How to make a Snake submission](how-to/how-to-make-a-snake-game-submission.md)
- [How to make an MNIST submission](how-to/how-to-make-an-mnist-submission.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the review flow. The
`Submission check` workflow validates paths and `config.json` without executing
the submitted model.

## Leaderboard

The site in `docs/` is deployed by `.github/workflows/pages.yml`:

https://wd7512.github.io/minimal-nn-competition/

Pages must use "GitHub Actions" as its source. To rebuild the site locally:

```bash
uv run mnc download-mnist --output data
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

`--replay-dir` records one seed-0 game per Snake submission. `--replays` renders
each game under `docs/games/`. Inspect a saved leaderboard without rewriting it:

```bash
uv run mnc leaderboard leaderboard.json
```

## Maintainer review

Submission PRs merge into `submission`. After evaluating an entry, commit the
updated leaderboard to `submission`, then merge `submission` into `main`.
Feature and maintainer changes target `dev` first.

```bash
uv run mnc evaluate <benchmark> models/<benchmark>/<name>
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full process.

## Internal token leaderboard

Score this repo's own code for syntactic compactness (keywords, operators, and
literals count; identifiers, comments, whitespace, and docstrings do not) across
the `core`, `mnist`, and `snake` scopes. A scope scores only when all of its
mapped tests pass:

```bash
uv run mnc internal
uv run mnc render-internal internal-leaderboard.json docs
```

The hidden page lands at `docs/internal/` alongside the speed board.

## Development checks

`mnc`, `minimal-nn-competition`, and `minimal_nn_competition` are equivalent
entry points; `python -m minimal_nn_competition` also works.

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```
