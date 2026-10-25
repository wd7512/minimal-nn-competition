"""A github pages leaderboard for the minimal neural network competition."""

import json
import statistics
from dataclasses import asdict

from minimal_nn_competition.benchmarks.chess_eval.names import CHESS_BENCHMARKS
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.runner.entry import BenchmarkResult


def _game_length(result):
    if result.benchmark not in SNAKE_BENCHMARKS:
        return 0.0
    moves = result.details.get("seed_moves")
    if not isinstance(moves, list) or not moves:
        return float("inf")
    return statistics.fmean(moves)


def _mean_error(result):
    if result.benchmark not in CHESS_BENCHMARKS:
        return 0.0
    mae = result.details.get("mae", 0.0)
    return mae if isinstance(mae, (int, float)) else 0.0


def rank_results(results):
    """Rank by passing status, score, parameter count, Snake game length, then Chess mean error."""
    # For the Snake competition it would be nice to have some renderings of the games.
    return sorted(
        results,
        key=lambda result: (
            result.benchmark,
            not result.passed,
            -result.score,
            result.parameter_count,
            _game_length(result),
            _mean_error(result),
            result.submission,
        ),
    )


def save_results(path, results):
    payload = [asdict(result) for result in rank_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_results(path):
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
                name=str(item.get("name", item.get("submission", "unknown"))),
                author=str(item.get("author", "unknown")),
            )
        )
    return results
