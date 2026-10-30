# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

## Submit a neural network
Rules: submissions use feed-forward `linear` or `mlp` models and a PyTorch `state_dict` with all tensors stored as `float32`.

1. Fork this repo
2. Open a pull request against `submission` with exactly
`models/<benchmark>/<name>/config.json` and `model.pt`.

Check your folder locally before submitting:

```bash
uv run mnnc validate-submission models/<benchmark>/<name>
```

See [validation and local leaderboard updates](how-to/how-to-validate-a-submission.md).
For the general submission workflow, see [how to make a submission](how-to/how-to-make-a-submission.md).

Submission Guides: [Snake](how-to/how-to-make-a-snake-submission.md), [Image compression](how-to/how-to-make-an-image-compression-submission.md), [Chess](how-to/how-to-make-a-chess-submission.md), [Control](how-to/how-to-make-a-control-submission.md)

## Control benchmarks

Controls use versioned Gymnasium physics and native recorded videos: `cartpole-gym-v1`,
`pendulum-gym-v1` (one continuous torque), and `lunarlander-gym-v1` (Box2D).
See the [evaluation contract and training commands](how-to/how-to-make-a-control-submission.md).
Install FFmpeg with libx264 to record episodes; rendering saved videos needs no encoder.

## Submit/Propose a new benchmark

Create an issue with our [benchmark template](https://github.com/wd7512/minimal-nn-competition/blob/main/.github/ISSUE_TEMPLATE/new-benchmark.md).
