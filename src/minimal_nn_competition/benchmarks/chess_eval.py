"""Chess evaluation: predict Stockfish's centipawn score within 0.1 pawns."""

import functools
import json
from pathlib import Path

from minimal_nn_competition.model.entry import materialize_submission

__all__ = [
    "CHESS_EVAL_DATA",
    "ChessEval10kBenchmark",
    "ChessEval100kBenchmark",
    "ChessEvalBenchmark",
    "INPUT_SIZE",
    "OUTPUT_SIZE",
    "TOLERANCE_PAWNS",
    "encode",
    "load_tensors",
]

INPUT_SIZE = 773
OUTPUT_SIZE = 1
TOLERANCE_PAWNS = 0.1
DEFAULT_FOLDS = 20
_DATA_DIRECTORY = Path(__file__).parent / "data" / "chess_eval"
CHESS_EVAL_DATA = {
    "chess-eval-10k": _DATA_DIRECTORY / "positions-10k.jsonl",
    "chess-eval-100k": _DATA_DIRECTORY / "positions-100k.jsonl",
}

_PIECES = {piece: index for index, piece in enumerate("PNBRQKpnbrqk")}


def encode(fen):
    """Encode a FEN as piece planes, side to move, and KQkq castling flags.

    Squares follow FEN order: a8=0, h8=7, a1=56, h1=63.
    Value 768 is +1 for White to move and -1 for Black.
    Values 769:773 indicate K, Q, k, q castling rights as zero/one flags.
    """
    parts = fen.split()
    inputs = [0.0] * (12 * 64)
    square = 0
    for symbol in parts[0]:
        if symbol == "/":
            continue
        if symbol.isdigit():
            square += int(symbol)
            continue
        inputs[_PIECES[symbol] * 64 + square] = 1.0
        square += 1
    inputs.append(1.0 if parts[1] == "w" else -1.0)
    inputs.extend(float(right in parts[2]) for right in "KQkq")
    return inputs


@functools.lru_cache(maxsize=2)
def _load_tensors(path):
    import torch

    positions = _load_positions(path)
    inputs = torch.zeros((len(positions), INPUT_SIZE), dtype=torch.float32)
    targets = torch.empty(len(positions), dtype=torch.float32)
    for row, (fen, cp) in enumerate(positions):
        inputs[row] = torch.tensor(encode(fen), dtype=torch.float32)
        targets[row] = cp / 100.0
    return inputs, targets


@functools.lru_cache(maxsize=4)
def _load_positions(path):
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Chess evaluation data is unavailable; expected the committed dataset at {data_path}"
        )
    positions = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        positions.append((str(item["fen"]), int(item["cp"])))
    if not positions:
        raise ValueError(f"Chess evaluation data is empty: {data_path}")
    return tuple(positions)


def load_tensors(data_path):
    """Return the evaluation inputs and pawn targets for ``data_path``."""
    return _load_tensors(str(Path(data_path)))


class ChessEvalBenchmark:
    name = ""
    default_data = None

    def __init__(self, data_path=None, folds=DEFAULT_FOLDS):
        if folds <= 0:
            raise ValueError("Chess evaluation folds must be positive")
        self.data_path = Path(data_path) if data_path is not None else self.default_data
        self.folds = folds

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Chess evaluation")
        if submission.config.input_size not in (769, INPUT_SIZE):
            raise ValueError(f"Chess evaluation requires input_size {INPUT_SIZE} or legacy 769")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError(f"Chess evaluation submissions must declare output_size {OUTPUT_SIZE}")
        return materialize_submission(submission)

    def evaluate(self, submission):
        import torch

        inputs, targets = load_tensors(self.data_path)
        with torch.no_grad():
            predictions = submission.model(inputs[:, : submission.config.input_size]).squeeze(1)
            errors = (predictions - targets).abs()
            within = errors <= TOLERANCE_PAWNS
        folds = min(self.folds, len(targets))
        fold_scores = [within[index::folds].float().mean().item() for index in range(folds)]
        return (
            within.float().mean().item(),
            bool(within.all().item()),
            {
                "samples": len(targets),
                "encoding": "castling-v2"
                if submission.config.input_size == INPUT_SIZE
                else "board-v1",
                "tolerance_pawns": TOLERANCE_PAWNS,
                "mae": errors.mean().item(),
                "folds": folds,
                "fold_scores": fold_scores,
            },
        )


class ChessEval10kBenchmark(ChessEvalBenchmark):
    name = "chess-eval-10k"
    default_data = CHESS_EVAL_DATA[name]


class ChessEval100kBenchmark(ChessEvalBenchmark):
    name = "chess-eval-100k"
    default_data = CHESS_EVAL_DATA[name]
