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

- [Snake Simple 8x8](how-to/how-to-make-a-snake-simple-8x8-submission.md)
- [Snake Simple 16x16](how-to/how-to-make-a-snake-simple-16x16-submission.md)
- [Snake Obstacles 8x8](how-to/how-to-make-a-snake-obstacles-8x8-submission.md)
- [MNIST](how-to/how-to-make-an-mnist-submission.md)
- [Fashion-MNIST](how-to/how-to-make-a-fashion-mnist-submission.md)
- [Chess evaluation](how-to/how-to-make-a-chess-eval-submission.md)
- [Control (RL)](how-to/how-to-make-an-rl-submission.md)

Snake Obstacles 8x8 seeds two or four blocked cells per run. A submission
passes only when every seed fills all free cells and then completes one extra
loop move; layouts are proved passable with a closed-tour search.

Maintainers: follow [How to validate a submission](how-to/how-to-validate-a-submission.md).

## Development

Code, docs, and workflow changes go through a pull request to `dev`, then
`main`. `main` and `submission` require pull requests; admins may bypass
protection when needed. Only submissions use the `submission` branch.

```bash
uv sync --dev --locked
uv run dev-tools check
```

Individual diagnostics are available with `uv run dev-tools tokens`,
`uv run dev-tools snake-speed`, and `uv run dev-tools char-encoding`.

## Hidden boards

The site also publishes the Snake Simple 8x8 speed and internal token boards:

```bash
uv run mnnc speed
uv run mnnc render-speed speed-leaderboard.json docs/internal
uv run mnnc internal
uv run mnnc render-internal internal-leaderboard.json docs
```

Pages deploys `docs/` on every push to `main`.
