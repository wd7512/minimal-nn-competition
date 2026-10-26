# How to make an Image compression submission

The Image compression group contains two image-classification benchmarks: MNIST
and Fashion-MNIST. Both use 784 flattened pixel inputs and 10 outputs. The score
is the fraction of all 70,000 examples (the official 60,000 training and 10,000
test images combined) classified correctly; a submission passes at `1.0` — every
example. There is no held-out split.

Use `models/mnist/<name>/` for MNIST and `models/fashion-mnist/<name>/` for
Fashion-MNIST.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

The directory name is the submission slug shown on the leaderboard.

## 2. Write `config.json`

MNIST:

```json
{
  "benchmark": "mnist",
  "name": "example",
  "author": "example author",
  "architecture": "mlp",
  "input_size": 784,
  "output_size": 10,
  "hidden_sizes": [64],
  "activation": "relu",
  "normalisation": false,
  "description": "A small MNIST classifier."
}
```

Fashion-MNIST:

```json
{
  "benchmark": "fashion-mnist",
  "name": "example",
  "author": "example author",
  "architecture": "mlp",
  "input_size": 784,
  "output_size": 10,
  "hidden_sizes": [64],
  "activation": "relu",
  "normalisation": false,
  "description": "A small Fashion-MNIST classifier."
}
```

- `name` and `author` are required and shown on the leaderboard.
- `description` is optional and is not shown on the site.
- `architecture` is `linear` or `mlp`.
- `input_size` must be `784` and `output_size` must be `10`.
- `normalisation` is optional and defaults to `false`.

`mlp` submissions declare `hidden_sizes` and `activation`. `linear` submissions
must omit `hidden_sizes`.

Set `"normalisation": true` to divide pixels by 255 before evaluation. The
setting must match how the model was trained. The checked-in examples use raw
`0-255` pixels.

## 3. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 4. Evaluate locally

Download the official dataset, then evaluate the example:

```bash
uv sync --dev --locked
uv run mnnc download-mnist --output data
uv run mnnc evaluate mnist models/mnist/example
```

For Fashion-MNIST:

```bash
uv run mnnc download-fashion-mnist --output data
uv run mnnc evaluate fashion-mnist models/fashion-mnist/example
```

The downloads write single `data/mnist.pt` and `data/fashion-mnist.pt` caches
with all 70,000 examples. The caches store raw `(inputs, labels)` tensors, with
inputs flattened to 784 values. The evaluator rejects caches that do not contain
exactly 70,000 samples.

## Scoring

Both benchmarks evaluate all 70,000 examples. Train on the same data you are
scored on and fit every example exactly. The score is the fraction classified
correctly, so a passing submission scores `1.0` (shown as `100.00%` on the
site). Results rank passing submissions before failing submissions, then by
descending score and ascending parameter count.

## Train the examples

The checked-in examples can be trained locally:

```bash
uv run mnnc download-mnist --output data
uv run mnnc train-mnist
uv run mnnc download-fashion-mnist --output data
uv run mnnc train-fashion-mnist
```

Training uses Apple MPS automatically when available and takes only a few
minutes. The examples train on all 70,000 examples using raw `0-255` pixels;
pass `--normalise` to train a normalised variant instead.

## 5. Open the pull request

Fork this repository on GitHub first. Then clone your fork and create a
short-lived submission branch from the integration branch:

```bash
git fetch origin
git switch -c submit/<name> origin/submission
```

Add only `config.json` and `model.pt`, commit, push the branch to your fork,
and open a pull request targeting the upstream `submission` branch. Include
only the two model files.

Target the `submission` branch and include only `config.json` and `model.pt`.
See [How to validate a submission](how-to-validate-a-submission.md) for the
review flow.
