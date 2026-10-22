# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

Problems: **Snake** (fill an 8×8 board) · **MNIST** (classify handwritten digits).
Rules: feed-forward `linear` or `mlp` models, a standard activation, and a PyTorch `state_dict`.

## Submit

Open a pull request against `submission` with exactly
`models/<benchmark>/<name>/config.json` and `model.pt`. Do not change code,
dependencies, workflows, the leaderboard, or `docs/`.

- [How to make a Snake submission](how-to/how-to-make-a-snake-game-submission.md)
- [How to make an MNIST submission](how-to/how-to-make-an-mnist-submission.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for leaderboard rebuilds, internal boards,
local development, and the review flow.
