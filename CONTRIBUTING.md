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

**Base submission branches on the latest `origin/submission`, not `main` or `dev`.**
The validator compares the PR against the integration branch so unrelated code and
leaderboard history cannot be mistaken for submission changes. Before opening a PR:

```bash
git fetch origin
git switch -c submit/<name> origin/submission
# add only the two model files, commit, and open a PR targeting submission
```

If `origin/submission` advances before your PR is merged, rebase your short-lived
branch onto the latest `origin/submission` and push the updated branch. Stale or
diverged PR heads are rejected with a remediation message instead of being merged
with an ambiguous file diff.

Benchmark guides:

- [Snake](how-to/how-to-make-a-snake-submission.md)
- [MNIST](how-to/how-to-make-an-mnist-submission.md)
- [Fashion-MNIST](how-to/how-to-make-a-fashion-mnist-submission.md)
- [Chess evaluation](how-to/how-to-make-a-chess-eval-submission.md)
- [Control (RL)](how-to/how-to-make-an-rl-submission.md)

Each submission's detail page shows its structure — for example
`[784,14,14,10]` — directly above the parameter count, taken from
`input_size`, `hidden_sizes`, and `output_size` in `config.json`.

Snake Obstacles 8x8 seeds two or four blocked cells per run. A submission
passes only when every seed fills all free cells and then completes one extra
loop move; layouts are proved passable with a closed-tour search.

Maintainers: follow [How to validate a submission](how-to/how-to-validate-a-submission.md).

## Development

Code, docs, and workflow changes go through a pull request to `dev`, then
`main`. `main`, `dev`, and `submission` require pull requests. `main` and `dev`
require the `test` check and an up-to-date branch; `submission` requires the
`validate` check. Admins may bypass protection when needed. Only submissions
use the `submission` branch.

Never enable automatic branch deletion, and never use
`gh pr merge --delete-branch` when the PR head is `main`, `dev`, or
`submission`. Delete only short-lived `submit/*` and feature branches.

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
uv run mnnc render-speed docs/internal/speed/leaderboard.json docs/internal
uv run mnnc internal
uv run mnnc render-internal docs/internal/leaderboard.json docs
```

Pages deploys `docs/` on every push to `main`.
