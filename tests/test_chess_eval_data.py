import hashlib
import json
from collections import Counter

from minimal_nn_competition.benchmarks.chess_eval.benchmark import CHESS_EVAL_DATA, encode

TIERS = (("10k", 10_000), ("100k", 100_000))


def _rows(name):
    path = CHESS_EVAL_DATA[f"chess-eval-{name}"]
    metadata = json.loads(path.with_name(f"metadata-{name}.json").read_text())
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return path, metadata, rows


def test_chess_eval_dataset_integrity() -> None:
    for name, positions in TIERS:
        path, metadata, rows = _rows(name)
        assert len(rows) == positions == metadata["positions"]
        assert len({row["fen"] for row in rows}) == len(rows)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        assert metadata["license"] == "CC0-1.0"
        assert metadata["input"]["size"] == 769
        assert metadata["tolerance_pawns"] == 0.1
        assert all(abs(row["cp"]) <= metadata["filters"]["max_abs_cp"] for row in rows)
        bands = Counter(
            "black" if row["cp"] < -50 else "white" if row["cp"] > 50 else "draw" for row in rows
        )
        assert dict(bands) == metadata["bands"]
        assert all(row["fen"].count("/") == 7 for row in rows)
        assert all(len(encode(row["fen"])) == 769 for row in rows[:1_000])


def test_chess_eval_100k_contains_10k() -> None:
    _, metadata, rows = _rows("100k")
    _, _, ten_k_rows = _rows("10k")
    assert metadata["superset_of"] == "chess-eval-10k"
    assert {row["fen"] for row in ten_k_rows} <= {row["fen"] for row in rows}
