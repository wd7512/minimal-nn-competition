# How to make a Chess evaluation submission

Chess evaluation predicts Stockfish's position score from the board. The score
is a single number in pawns from White's point of view. A position is correct
when the prediction is within 0.1 pawns of Stockfish. A submission passes only
when every position in the fixed evaluation set is correct.

There are two evaluation sets:

| Benchmark | Positions | Use |
| --- | --- | --- |
| `chess-eval-10k` | 10,000 | The standard tier |
| `chess-eval-100k` | 100,000 | The larger capacity tier; includes every 10k position |

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/chess-eval-10k/<name>/config.json
models/chess-eval-10k/<name>/model.pt
```

Use `models/chess-eval-100k/<name>/` for the larger tier. The directory name is
the submission slug shown on the leaderboard.

## 2. Write `config.json`

```json
{
  "benchmark": "chess-eval-10k",
  "name": "example",
  "author": "example author",
  "architecture": "mlp",
  "input_size": 769,
  "output_size": 1,
  "hidden_sizes": [64, 64],
  "activation": "relu",
  "description": "A small chess evaluation network."
}
```

- `benchmark` must be `chess-eval-10k` or `chess-eval-100k` and match the
  submission directory.
- `name` and `author` are required and shown on the leaderboard.
- `description` is optional and is not shown on the site.
- `architecture` is `linear` or `mlp`.
- `input_size` must be `769` and `output_size` must be `1`.
- `normalisation` is not used by this benchmark.
- `linear` submissions must omit `hidden_sizes`; `mlp` submissions declare
  `hidden_sizes` and `activation`.

## 3. Encode the position

The input is 769 values. Both tiers use the same encoding:

- 12 piece planes of 64 squares each, in this order:
  `white P`, `white N`, `white B`, `white R`, `white Q`, `white K`,
  `black P`, `black N`, `black B`, `black R`, `black Q`, `black K`.
  Each square is `1.0` when that piece is present and `0.0` otherwise.
- One final value: `+1.0` when White is to move, `-1.0` when Black is to move.

Squares follow FEN order: `a8=0`, `h8=7`, `a1=56`, `h1=63`. Castling rights and
en passant are not inputs; positions with an en-passant capture are excluded
from the evaluation sets.

## 4. Output and scoring

The model must output one number per position: Stockfish's score in pawns from
White's point of view (`cp / 100`). The score is the fraction of positions where
`abs(prediction - target) <= 0.1`.

A submission passes only at `1.0` (every position within 0.1 pawns). Results
rank passing submissions first, then by descending score, ascending parameter
count, and ascending mean error.

## 5. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint. All tensors in the state dict must use `float32`.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 6. Evaluate locally

The evaluation sets are committed with the benchmark, so no download is needed:

```bash
uv sync --dev --locked
uv run mnnc evaluate chess-eval-10k models/chess-eval-10k/example
uv run mnnc evaluate chess-eval-100k models/chess-eval-100k/example
```

The checked-in 10k example is a `769-64-64-1` ReLU MLP with 53,505 parameters.
It scores `1.0` and passes. The checked-in 100k example reuses those weights as
a transfer baseline; it fails the larger tier, which is expected.

## 7. Dataset

The fixed evaluation sets use deduplicated positions from the
[Lichess CC0 evaluation database](https://database.lichess.org/). Each position
keeps the highest-depth Stockfish evaluation with at least 30 plies and
500,000 knodes, has no forced mate, no en-passant capture, and a centipawn
score within ±10 pawns.

- `chess-eval-10k` contains 10,000 positions: 3,333 with `cp < -50`,
  3,334 with `-50 <= cp <= 50`, and 3,333 with `cp > 50`.
- `chess-eval-100k` contains 100,000 positions: 33,333 with `cp < -50`,
  33,334 with `-50 <= cp <= 50`, and 33,333 with `cp > 50`. It is a strict
  superset of the 10k tier, so the 10k positions keep the same labels.

The datasets and their provenance metadata live in
`src/minimal_nn_competition/benchmarks/data/chess_eval/`. Maintainers can rebuild
them with:

```bash
uv run --with duckdb python scripts/build_chess_eval_dataset.py --tier both
```

There is no held-out split: train on the same positions you are scored on and
fit every position within 0.1 pawns.

## Train the examples

The checked-in 10k example can be trained locally:

```bash
uv run mnnc train-chess-eval-10k
```

The default run trains a `769-64-64-1` ReLU MLP with MSE, Adam, cosine learning
rate decay, and deterministic shuffling. It stops once every position is within
0.1 pawns.

To train on the larger tier, select its command and output directory:

```bash
uv run mnnc train-chess-eval-100k --output models/chess-eval-100k/example
```

Training the 100k tier takes longer and needs a larger network than the 10k
example to pass.

## 8. Open the pull request

Fork this repository on GitHub first. Clone your fork, then add the upstream
remote once:

```bash
git remote add upstream https://github.com/wd7512/minimal-nn-competition.git
git fetch upstream
git switch -c submit/<name> upstream/submission
```

`origin` stays your fork; push the branch there:

```bash
git push -u origin submit/<name>
```

Open a pull request targeting the upstream `submission` branch. Keep
`config.json` and `model.pt` as the only changed files. See
[How to validate a submission](how-to-validate-a-submission.md) for the
review flow.
