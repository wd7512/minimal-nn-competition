# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

## Submit a neural network
Rules: submissions use feed-forward `linear` or `mlp` models and a PyTorch `state_dict` with all tensors stored as `float32`.

1. Fork this repo
2. Open a pull request against `submission` with exactly
`models/<benchmark>/<name>/config.json` and `model.pt`.

Submission Guides: [Snake](how-to/how-to-make-a-snake-submission.md), [Image compression](how-to/how-to-make-an-image-compression-submission.md), [Chess](how-to/how-to-make-a-chess-submission.md), [Control](how-to/how-to-make-a-control-submission.md)

## Submit/Propose a new benchmark

Create an issue with our [benchmark template](https://github.com/wd7512/minimal-nn-competition/blob/main/.github/ISSUE_TEMPLATE/new-benchmark.md).
