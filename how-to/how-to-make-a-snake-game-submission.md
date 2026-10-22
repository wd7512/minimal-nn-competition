# How to make a Snake submission

Snake uses an 8×8 board. A model observes 24 features and chooses one of four
directions. A submission passes when it fills all 64 cells on every seed.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/snake/<name>/config.json
models/snake/<name>/model.pt
```

The directory name is the submission slug shown on the leaderboard.

## 2. Write `config.json`

```json
{
  "benchmark": "snake",
  "name": "example",
  "author": "example author",
  "architecture": "mlp",
  "input_size": 24,
  "output_size": 4,
  "hidden_sizes": [64, 64],
  "activation": "relu",
  "description": "A small PPO-trained MLP policy."
}
```

- `name` and `author` are required and shown on the leaderboard.
- `description` is optional and is not shown on the site.
- `architecture` is `linear` or `mlp`.
- `input_size` is `24`, or the length of `input_indices`.
- `output_size` must be `4`.

`mlp` submissions declare `hidden_sizes` and `activation`. `linear` submissions
must omit `hidden_sizes`.

### Selecting observation features

The 24 features cover head position, distances to walls and corners, food
alignment and diagonals, and the nearest body segment in each of eight
directions. Set `input_indices` to use a subset; `input_size` must match its
length. Omit it or set it to `true` to use all 24.

```json
{
  "benchmark": "snake",
  "name": "example",
  "author": "example author",
  "architecture": "linear",
  "input_size": 5,
  "output_size": 4,
  "input_indices": [0, 1, 2, 4, 7]
}
```

## 3. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 4. Evaluate locally

```bash
uv sync --dev --locked
uv run mnc evaluate snake models/snake/example
```

## Scoring

Snake evaluates each model across 100 seeds (`0`–`99`) with a 4096-move cap.
The score is the mean final snake length as a percentage of the 64-cell board,
and the site also shows the standard deviation across seeds. The `Game length`
column shows the mean number of ticks per game and its standard deviation.

Results rank passing submissions before failing submissions, then by descending
score and ascending parameter count. Snake submissions with equal score and
parameter count are ranked by ascending game length, so faster games win.

## Train the example

The checked-in example can be trained locally:

```bash
uv run mnc train-snake
```

Training uses Apple MPS automatically when available and takes only a few
minutes.

## 5. Open the pull request

Target the `submission` branch and include only `config.json` and `model.pt`.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the review flow.
