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
- Open a PR against the `submission` branch with your model under `models/<benchmark>/<name>/`
- The PR must contain only `config.json` and `model.pt`; do not include code, dependency, or workflow changes
- After the PR is merged into `submission`, I evaluate it locally on this machine and add it to the leaderboard
- Accepted submissions are then merged into `main`; feature and maintainer changes target `dev` first

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full submission and review flow.
The `Submission check` workflow enforces the two-file rule without executing the
submitted model.

## Submission format

A submission is a directory containing `config.json` and `model.pt`:

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

`name` is the display name shown on the leaderboard and `author` credits
who made the submission. `description` is optional: include a short paragraph
(or less) explaining how you built the network or its style. It is not shown
on the leaderboard site.

MNIST submissions may also set `"normalisation": true` to scale pixels from
`0-255` to `0-1`. The example MNIST model is trained on non-normalised inputs,
so its config uses `false`. For example:

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
  "normalisation": false
}
```

Snake uses a 24-feature observation adapted from the Snake-Learning project.
The features cover head position and distances to the walls and corners, food
alignment/diagonals, and the nearest body segment in each of eight directions.
To use only some of them, set `input_indices`; `input_size` must match its
length. Omit it or set it to `true` to use all 24:

```json
{
  "benchmark": "snake",
  "architecture": "linear",
  "input_size": 5,
  "output_size": 4,
  "input_indices": [0, 1, 2, 4, 7]
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

MNIST scores are full-test-set accuracy, shown as a percentage; a submission
passes at 100% accuracy. Snake uses an 8x8 board and evaluates each model across
100 seeds (`0`–`99`) with a 4096-move episode cap. Its score is the mean final
snake length as a percentage of the 64-cell board, and the site also shows the
standard deviation across seeds. The `Game length` column shows the mean number
of ticks per game and its standard deviation across seeds. A submission passes
only when it fills the board (length 64) on every seed.

Results rank passing submissions before failing submissions, then by descending
score and ascending parameter count. Submit a pull request against `submission`
containing only the model directory under `models/<benchmark>/<name>/`.

## Review process

Submission PRs target `submission` and contain only `config.json` and `model.pt`.
After merging an entry into `submission`, the maintainer evaluates it locally:

```bash
uv run mnc evaluate <benchmark> models/<benchmark>/<name>
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

The updated leaderboard is committed to `submission`, then `submission` is
merged into `main`. GitHub Pages deploys from `main` only. Feature and
maintainer changes target `dev` and are promoted to `main` when ready.

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
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

`mnc collect` evaluates every `models/<benchmark>/<name>/` directory.
`--replay-dir` records one seed-0 game per snake submission; `--replays`
renders each as a `games/<name>.html` page linked from the snake table
with a "view game" link. To inspect
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

## Internal token leaderboard

Score the repo's own code for syntactic compactness (keywords, operators,
and literals count; identifiers, comments, whitespace, and docstrings do
not) across the `core`, `mnist`, and `snake` scopes. A scope scores only
when all of its mapped tests pass:

```bash
uv run mnc internal
uv run mnc render-internal internal-leaderboard.json docs
```

The hidden page lands at `docs/internal/` alongside the speed board.

## Development checks

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```
