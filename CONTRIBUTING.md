# Contributing

## Submissions

Submit a model by opening a pull request against `submission`, not `main`.
Include exactly two files in a new directory:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

Do not change code, dependencies, workflows, the leaderboard, or `docs/`.
`config.json` must include non-empty `name` and `author` strings; `description`
is optional.

Benchmark guides:

- [Snake](how-to/how-to-make-a-snake-game-submission.md)
- [MNIST](how-to/how-to-make-an-mnist-submission.md)
- [Fashion-MNIST](how-to/how-to-make-a-fashion-mnist-submission.md)

Maintainers: follow [How to validate a submission](how-to/how-to-validate-a-submission.md).

## Development

Code, docs, and workflow changes go through a pull request to `dev`, then
`main`. `main` and `submission` require pull requests; admins may bypass
protection when needed. Only submissions use the `submission` branch.

```bash
uv sync --dev --locked
uv run mnc list
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## Hidden boards

The site also publishes the Snake speed and internal token boards:

```bash
uv run mnc speed
uv run mnc render-speed speed-leaderboard.json docs/internal
uv run mnc internal
uv run mnc render-internal internal-leaderboard.json docs
```

Pages deploys `docs/` on every push to `main`.
