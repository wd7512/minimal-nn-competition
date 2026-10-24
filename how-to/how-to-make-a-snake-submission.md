# How to make a Snake submission

The Snake group contains three fixed-seed benchmarks: Snake Simple 8x8, Snake
Simple 16x16, and Snake Obstacles 8x8. Each submission is a model that observes
24 features and chooses one of four directions. No data files are needed.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

Use `snake-simple-8x8`, `snake-simple-16x16`, or `snake-obstacles-8x8` as the
benchmark directory. The submission directory name is the slug shown on the
leaderboard.

## 2. Write `config.json`

```json
{
  "benchmark": "snake-simple-8x8",
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
directions (the nearest body or blocked cell on Snake Obstacles 8x8). Set
`input_indices` to use a subset; `input_size` must match its length. Omit it or
set it to `true` to use all 24.

```json
{
  "benchmark": "snake-simple-8x8",
  "name": "example",
  "author": "example author",
  "architecture": "linear",
  "input_size": 5,
  "output_size": 4,
  "input_indices": [0, 1, 2, 4, 7]
}
```

## 3. Boards and passability

Snake Simple 8x8 uses an 8×8 board and passes when the snake fills all 64 cells
on every seed and makes the extra move. Snake Simple 16x16 uses a 16×16 board
and passes the same way on all 256 cells.

Snake Obstacles 8x8 seeds two or four blocked cells per run, never in the
centre. Each seed deterministically selects the blocked cells as one or two
opposite-colour pairs, resampled from the same RNG stream until the free cells
admit a closed Hamiltonian tour: starting at the centre, visiting every free
cell exactly once, and ending next to the centre. Some seeds therefore have 62
free cells and some have 60.

The game does not end when the last free cell is filled on any benchmark. The
snake must make one more successful move into the tail that is vacating;
colliding on that move loses the seed.

## 4. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 5. Scoring

Every benchmark evaluates 100 fixed seeds (`0`–`99`). The score is the fraction
of seeds that fill the board (every free cell on Snake Obstacles 8x8) and make
the extra move, so a passing submission scores `1.0`. The `Game length` column
shows the mean number of ticks per game and its standard deviation. Obstacle
detail pages show each seed's length as a percentage of that seed's free-cell
target.

Results rank passing submissions first, then by descending score (fraction of
seeds passed) and ascending parameter count. Submissions with equal score and
parameter count are ranked by ascending game length, so faster games win.

## 6. Evaluate locally

```bash
uv sync --dev --locked
uv run mnnc evaluate snake-simple-8x8 models/snake-simple-8x8/example
uv run mnnc evaluate snake-simple-16x16 models/snake-simple-16x16/example
uv run mnnc evaluate snake-obstacles-8x8 models/snake-obstacles-8x8/example
```

Validate every obstacle layout independently:

```bash
uv run mnnc validate-snake-obstacles-8x8 --seeds 100
```

The validator re-checks each tour and simulates a tour-following policy with the
same seed and blocked cells. It exits non-zero if any seed cannot fill every free
cell and complete the extra loop move within the 4096-move cap.

## Train the examples

The checked-in examples can be trained locally:

```bash
uv run mnnc train-snake-simple-8x8
uv run mnnc train-snake-simple-16x16
uv run mnnc train-snake-obstacles-8x8
```

Training uses Apple MPS automatically when available and takes only a few
minutes. Obstacle training episodes use the benchmark's blocked cells and
full-board rule, so a policy can learn the extra loop move.

## 7. Open the pull request

Target the `submission` branch and include only `config.json` and `model.pt`.
See [How to validate a submission](how-to-validate-a-submission.md) for the
review flow.
