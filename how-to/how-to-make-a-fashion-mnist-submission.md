# How to make a Fashion-MNIST submission

Fashion-MNIST uses 784 flattened pixel inputs and 10 outputs. The score is
accuracy on all 70,000 Fashion-MNIST examples (the official 60,000 training and
10,000 test images combined); a submission passes at 100%. There is no held-out
split.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/fashion-mnist/<name>/config.json
models/fashion-mnist/<name>/model.pt
```

The directory name is the submission slug shown on the leaderboard.

## 2. Write `config.json`

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
setting must match how the model was trained. The checked-in example uses raw
`0-255` pixels.

## 3. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 4. Evaluate locally

Download the official Fashion-MNIST dataset, then evaluate the example:

```bash
uv sync --dev --locked
uv run mnnc download-fashion-mnist --output data
uv run mnnc evaluate fashion-mnist models/fashion-mnist/example
```

The download writes a single `data/fashion-mnist.pt` cache with all 70,000
examples. The cache stores raw `(inputs, labels)` tensors, with inputs flattened
to 784 values. The evaluator rejects caches that do not contain exactly 70,000
samples.

## Scoring

Fashion-MNIST evaluates all 70,000 examples. Train on the same data you are
scored on and fit every example exactly. The score is accuracy shown as a
percentage; a submission passes at 100%. Results rank passing submissions before
failing submissions, then by descending score and ascending parameter count.

## Train the example

The checked-in example can be trained locally:

```bash
uv run mnnc download-fashion-mnist --output data
uv run mnnc train-fashion-mnist
```

Training uses Apple MPS automatically when available and takes only a few
minutes. The example trains on all 70,000 examples using raw `0-255` pixels;
pass `--normalise` to train a normalised variant instead.

## 5. Open the pull request

Target the `submission` branch and include only `config.json` and `model.pt`.
See [How to validate a submission](how-to-validate-a-submission.md) for the
review flow.
