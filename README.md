# minimal-nn-competition

A platform for presenting and finding the smallest possible neural-network for problems.

Current problems
- Playing Snake
- MNIST classification

Rules
- Feed forward MLP
- ReLU activation function or activation function
- Pytorch models

How to submit
- PR into this repo with your model under `models/<benchmark>/<name>/`
- I will run this and add it to the leaderboard

## Submission format

A submission is a directory containing `config.json` and `model.pt`:

```json
{
  "benchmark": "snake",
  "architecture": "mlp",
  "input_size": 7,
  "output_size": 4,
  "hidden_sizes": [64, 64],
  "activation": "relu"
}
```

MNIST submissions may also set `"normalisation": true` to scale pixels from
`0-255` to `0-1`. The example MNIST model is trained on non-normalised inputs,
so its config uses `false`. For example:

```json
{
  "benchmark": "mnist",
  "architecture": "mlp",
  "input_size": 784,
  "output_size": 10,
  "hidden_sizes": [64],
  "activation": "relu",
  "normalisation": false
}
```

`architecture` is either `linear` or `mlp`. Linear checkpoints use the usual
`weight` and `bias` keys. MLP checkpoints use the keys produced by
`torch.nn.Sequential` (`0.weight`, `0.bias`, `2.weight`, ...), with ReLU layers
between linear layers.

`model.pt` must be a PyTorch state dict. The evaluator loads checkpoints with
`weights_only=True`; it does not execute arbitrary Python submitted with a
checkpoint.

## Local evaluation

```bash
uv sync --dev --locked
uv run mnc list
uv run mnc evaluate snake models/snake/example
uv run mnc evaluate mnist models/mnist/example --data-path data/mnist-test.pt
```

`mnc`, `minimal-nn-competition`, and `minimal_nn_competition` are equivalent
installed entry points. `python -m minimal_nn_competition` also works.

Download the official MNIST caches first:

```bash
uv run mnc download-mnist --output data
```

The cache stores raw `(inputs, labels)` tensors. Inputs are flattened to 784
values. If `normalisation` is enabled, the evaluator divides them by 255.

MNIST scores are full-test-set accuracy; a submission passes at 100% accuracy.
Snake uses an 8x8 board and evaluates each model across seeds `0`, `1`, and `2`
with a 4096-move episode cap. Its score is the average final snake length. A
submission passes only when it fills the board (length 64) on every seed.

Results rank passing submissions before failing submissions, then by descending
score and ascending parameter count. Submit a pull request with the model
directory under `models/<benchmark>/<name>/`.

## Example submissions

`src/minimal_nn_competition/examples/mnist_submission.py` and
`snake_submission.py` are runnable training examples. They generate the
checked-in submissions under `models/`:

```bash
uv run mnc download-mnist --output data
uv run mnc train-mnist
uv run mnc train-snake
```

Both trainers use Apple MPS automatically when available. Training the examples
takes only a few minutes. The MNIST example is trained on raw `0-255` pixels;
pass `--normalise` to train a normalised variant instead.

## Leaderboard and GitHub Pages

Evaluate all checked-in submissions and rebuild `leaderboard.json`:

```bash
uv run mnc download-mnist --output data
uv run mnc collect
uv run mnc render leaderboard.json docs
```

`mnc collect` evaluates every `models/<benchmark>/<name>/` directory. To inspect
a saved leaderboard without changing it:

```bash
uv run mnc leaderboard leaderboard.json
```

The checked-in `docs/` directory is deployed to GitHub Pages by
`.github/workflows/pages.yml`. The leaderboard includes a `Normalisation` column
for MNIST submissions. The site is published at
https://wd7512.github.io/minimal-nn-competition/. Enable Pages with
"GitHub Actions" as the source in the repository settings if it is not already
enabled.

## Development checks

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```
