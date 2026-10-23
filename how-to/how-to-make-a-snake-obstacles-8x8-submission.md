# How to make a Snake Obstacles 8x8 submission

Snake Obstacles 8x8 uses an 8×8 board with either two or four blocked cells on every
seed. The centre start cell is never blocked. A model observes 24 features and chooses
one of four directions. A submission passes when it fills every free cell on every seed
and then completes one extra loop move into the vacating tail.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/snake-obstacles-8x8/<name>/config.json
models/snake-obstacles-8x8/<name>/model.pt
```

The directory name is the submission slug shown on the leaderboard.

## 2. Write `config.json`

```json
{
  "benchmark": "snake-obstacles-8x8",
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
alignment and diagonals, and the nearest body or blocked cell in each of eight
directions. Set `input_indices` to use a subset; `input_size` must match its
length. Omit it or set it to `true` to use all 24.

```json
{
  "benchmark": "snake-obstacles-8x8",
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

## Obstacles and passability

Each seed deterministically selects two or four blocked cells away from the
centre, sampled as one or two opposite-colour pairs. A seed is passable when the
free cells admit a closed Hamiltonian tour:
starting at the centre, visiting every free cell exactly once, and ending next to
the centre. The project generates each layout with a bounded backtracking search
and resamples from the same RNG stream until an actual closed tour is found.

Some seeds therefore have 62 free cells and some have 60. The game does not end
when the last free cell is filled. The snake must make one more successful move
into the tail that is vacating; colliding on that move loses the seed.

## 4. Evaluate locally

```bash
uv sync --dev --locked
uv run mnnc evaluate snake-obstacles-8x8 models/snake-obstacles-8x8/example
```

Validate every generated layout independently:

```bash
uv run mnnc validate-snake-obstacles-8x8 --seeds 100
```

The validator re-checks each tour and simulates a tour-following policy with the
same seed and blocked cells. It exits non-zero if any seed cannot fill every free
cell and complete the extra loop move within the 4096-move cap.

## Scoring

Snake Obstacles 8x8 evaluates each model across 100 seeds (`0`–`99`) with a
4096-move cap. The score is the mean final snake length. The site shows each
seed's length as a percentage of that seed's free-cell target, then displays the
mean and population standard deviation across seeds. The `Game length` column
shows the mean number of ticks per game and its standard deviation.

A run passes only when every seed both reaches its target length and completes the
extra loop move. Results rank passing submissions before failing submissions,
then by descending score and ascending parameter count. Submissions with equal
score and parameter count are ranked by ascending game length, so faster games
win.

## Train the example

The example can be trained locally:

```bash
uv run mnnc train-snake-obstacles-8x8
```

Training uses the benchmark's blocked cells and the extra loop move on each
episode. It uses Apple MPS automatically when available.

## 5. Open the pull request

Target the `submission` branch and include only `config.json` and `model.pt`.
See [How to validate a submission](how-to-validate-a-submission.md) for the
review flow.
