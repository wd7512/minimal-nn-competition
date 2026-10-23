# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

Problems: **Snake Simple 8x8** and **Snake Simple 16x16** (fill the board) · **MNIST** and **Fashion-MNIST** (classify images).
Rules: feed-forward `linear` or `mlp` models, a standard activation, and a PyTorch `state_dict`.

MNIST and Fashion-MNIST each use one 70,000-example dataset. There is no
held-out split: train on the same examples you are scored on and fit every
example exactly.

## Submit

Open a pull request against `submission` with exactly
`models/<benchmark>/<name>/config.json` and `model.pt`. Do not change code,
dependencies, workflows, the leaderboard, or `docs/`.

- [How to make a Snake Simple 8x8 submission](how-to/how-to-make-a-snake-simple-8x8-submission.md)
- [How to make a Snake Simple 16x16 submission](how-to/how-to-make-a-snake-simple-16x16-submission.md)
- [How to make an MNIST submission](how-to/how-to-make-an-mnist-submission.md)
- [How to make a Fashion-MNIST submission](how-to/how-to-make-a-fashion-mnist-submission.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and
[how to validate a submission](how-to/how-to-validate-a-submission.md).
