# How to make a Chess evaluation submission

Chess evaluation predicts Stockfish's position score from the board. The score
is a single number in pawns from White's point of view. A position is correct
when the prediction is within 0.1 pawns of Stockfish. A submission passes only
when every position in the fixed evaluation set is correct.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/chess-eval/<name>/config.json
models/chess-eval/<name>/model.pt
```

The directory name is the submission slug shown on the leaderboard.

## 2. Write `config.json`

```json
{
  "benchmark": "chess-eval",
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

- `name` and `author` are required and shown on the leaderboard.
- `description` is optional and is not shown on the site.
- `architecture` is `linear` or `mlp`.
- `input_size` must be `769` and `output_size` must be `1`.
- `normalisation` is not used by this benchmark.
- `linear` submissions must omit `hidden_sizes`; `mlp` submissions declare
  `hidden_sizes` and `activation`.

## 3. Encode the position

The input is 769 values:

- 12 piece planes of 64 squares each, in this order:
  `white P`, `white N`, `white B`, `white R`, `white Q`, `white K`,
  `black P`, `black N`, `black B`, `black R`, `black Q`, `black K`.
  Each square is `1.0` when that piece is present and `0.0` otherwise.
- One final value: `+1.0` when White is to move, `-1.0` when Black is to move.

Squares follow FEN order: `a8=0`, `h8=7`, `a1=56`, `h1=63`. Castling rights and
en passant are not inputs; positions with an en-passant capture are excluded
from the evaluation set.

## 4. Output and scoring

The model must output one number per position: Stockfish's score in pawns from
White's point of view (`cp / 100`). The score is the fraction of positions where
`abs(prediction - target) <= 0.1`.

A submission passes only at `1.0` (every position within 0.1 pawns). Results
rank passing submissions first, then by descending score and ascending
parameter count.

## 5. Save `model.pt`

`model.pt` must be a PyTorch state dict. The evaluator loads it with
`weights_only=True`; it does not execute arbitrary Python submitted with the
checkpoint.

- `linear` checkpoints use `weight` and `bias`.
- `mlp` checkpoints use the keys produced by `torch.nn.Sequential` (`0.weight`,
  `0.bias`, `2.weight`, ...), with ReLU layers between linear layers.

## 6. Evaluate locally

The evaluation set is committed with the benchmark, so no download is needed:

```bash
uv sync --dev --locked
uv run mnnc evaluate chess-eval models/chess-eval/example
```

The checked-in example is a `769-64-64-1` ReLU MLP with 53,505 parameters. It
scores `1.0` and passes.

## 7. Dataset

The fixed evaluation set contains 10,000 deduplicated positions from the
[Lichess CC0 evaluation database](https://database.lichess.org/). Each position
keeps the highest-depth Stockfish evaluation with at least 30 plies and
500,000 knodes, has no forced mate, no en-passant capture, and a centipawn
score within ±10 pawns. The three bands are balanced:

- `3,333` positions with `cp < -50`
- `3,334` positions with `-50 <= cp <= 50`
- `3,333` positions with `cp > 50`

The dataset and its provenance metadata live in
`src/minimal_nn_competition/benchmarks/chess_eval/`. Maintainers can rebuild
the snapshot with:

```bash
uv run --with duckdb python scripts/build_chess_eval_dataset.py
```

There is no held-out split: train on the same positions you are scored on and
fit every position within 0.1 pawns.

## Train the example

The checked-in example can be trained locally:

```bash
uv run mnnc train-chess-eval
```

The default run trains a `769-64-64-1` ReLU MLP with MSE, Adam, cosine learning
rate decay, and deterministic shuffling. It stops once every position is within
0.1 pawns.
