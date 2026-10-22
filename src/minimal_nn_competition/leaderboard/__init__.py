"""A github pages leaderboard for the minimal neural network competition."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from minimal_nn_competition.runner.entry import BenchmarkResult


def rank_results(results: Iterable[BenchmarkResult]) -> list[BenchmarkResult]:
    """Rank by passing status, score, then smallest model within each benchmark."""
    # For the snake competition it would be nice to have some renderings of the games.
    return sorted(
        results,
        key=lambda result: (
            result.benchmark,
            not result.passed,
            -result.score,
            result.parameter_count,
            result.submission,
        ),
    )


def save_results(path: Path, results: Iterable[BenchmarkResult]) -> None:
    payload = [asdict(result) for result in rank_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_results(path: Path) -> list[BenchmarkResult]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Leaderboard must contain a JSON list")
    results = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Leaderboard entries must be JSON objects")
        results.append(
            BenchmarkResult(
                submission=str(item.get("submission", "unknown")),
                benchmark=str(item["benchmark"]),
                score=float(item["score"]),
                passed=bool(item["passed"]),
                parameter_count=int(item["parameter_count"]),
                details=dict(item["details"]),
            )
        )
    return results
