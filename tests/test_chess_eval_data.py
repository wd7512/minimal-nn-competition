import hashlib
import json
from collections import Counter, defaultdict

from minimal_nn_competition.benchmarks.chess_eval import CHESS_EVAL_DATA, encode

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
        assert metadata["input"]["size"] == 773
        assert metadata["tolerance_pawns"] == 0.1
        assert all(abs(row["cp"]) <= metadata["filters"]["max_abs_cp"] for row in rows)
        bands = Counter(
            "black" if row["cp"] < -50 else "white" if row["cp"] > 50 else "draw" for row in rows
        )
        assert dict(bands) == metadata["bands"]
        assert all(row["fen"].count("/") == 7 for row in rows)
        assert all(len(encode(row["fen"])) == 773 for row in rows[:1_000])


def test_chess_eval_100k_contains_10k() -> None:
    _, metadata, rows = _rows("100k")
    _, _, ten_k_rows = _rows("10k")
    assert metadata["superset_of"] == "chess-eval-10k"
    assert {row["fen"] for row in ten_k_rows} <= {row["fen"] for row in rows}


def test_castling_encoding_removes_incompatible_dataset_collisions() -> None:
    for name, _ in TIERS:
        _, _, rows = _rows(name)
        legacy = defaultdict(list)
        current = defaultdict(list)
        for row in rows:
            # Compact exact keys; values are -1, 0, or 1, so no information is lost.
            key = bytes(int(value + 1) for value in encode(row["fen"]))
            legacy[key[:769]].append(row["cp"])
            current[key].append(row["cp"])

        # A single prediction can cover a 20 cp interval at the ±10 cp tolerance.
        def unavoidable_errors(groups):
            return sum(
                len(scores) - max(sum(low <= cp <= low + 20 for cp in scores) for low in scores)
                for scores in groups.values()
            )

        if name == "100k":
            assert len(legacy) == 99_968
            assert sum(max(scores) - min(scores) > 20 for scores in legacy.values()) == 14
            assert unavoidable_errors(legacy) == 14
        assert unavoidable_errors(current) == 0


def test_checked_in_chess_examples_pass() -> None:
    from pathlib import Path

    from minimal_nn_competition.benchmarks.registry import create_benchmark
    from minimal_nn_competition.model.entry import Submission
    from minimal_nn_competition.runner.entry import evaluate_submission

    for tier, count in TIERS:
        name = f"chess-eval-{tier}"
        example = Path(f"models/{name}/example")
        submission = Submission.from_directory(example)
        assert submission.config.input_size == 773
        result = evaluate_submission(submission, create_benchmark(name))
        assert result.passed and result.score == 1.0
        assert result.details["samples"] == count
        assert result.details["encoding"] == "castling-v2"
