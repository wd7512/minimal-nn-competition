import hashlib
import json
from collections import Counter

from minimal_nn_competition.benchmarks.chess_eval.benchmark import DEFAULT_DATA, encode


def test_chess_eval_dataset_integrity() -> None:
    metadata_path = DEFAULT_DATA.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    rows = [
        json.loads(line)
        for line in DEFAULT_DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 10_000 == metadata["positions"]
    assert len({row["fen"] for row in rows}) == len(rows)
    assert hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["license"] == "CC0-1.0"
    assert metadata["input"]["size"] == 769
    assert metadata["tolerance_pawns"] == 0.1
    assert all(abs(row["cp"]) <= metadata["filters"]["max_abs_cp"] for row in rows)
    bands = Counter(
        "black" if row["cp"] < -50 else "white" if row["cp"] > 50 else "draw" for row in rows
    )
    assert dict(bands) == metadata["bands"]
    assert all(row["fen"].count("/") == 7 for row in rows)
    assert all(len(encode(row["fen"])) == 769 for row in rows)
