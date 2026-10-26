"""A github pages leaderboard for the minimal neural network competition."""

import json
import statistics
from dataclasses import asdict

from minimal_nn_competition.benchmarks.common.names import CHESS_BENCHMARKS, SNAKE_BENCHMARKS
from minimal_nn_competition.runner.entry import BenchmarkResult


def short_commit(commit):
    return commit[:7] if commit != "unknown" else commit


def format_measured(measured_at):
    return measured_at.replace("T", " ").replace("+00:00", " UTC") or "-"


def _game_length(result):
    if result.benchmark not in SNAKE_BENCHMARKS:
        return 0.0
    moves = result.details.get("seed_moves")
    if not isinstance(moves, list) or not moves:
        return float("inf")
    return statistics.fmean(moves)


def _longest_game_first(result):
    """Negated game length for descending order; missing games rank last."""
    length = _game_length(result)
    return float("inf") if length == float("inf") else -length


def _mean_error(result):
    if result.benchmark not in CHESS_BENCHMARKS:
        return 0.0
    mae = result.details.get("mae", 0.0)
    return mae if isinstance(mae, (int, float)) else 0.0


def rank_results(results):
    """Rank by passing status, score, parameter count, Snake game length, then Chess mean error.

    Failed Snake entries instead rank by longest game first, then
    parameter count, since survival is the only signal when nothing is won.
    """
    # For the Snake competition it would be nice to have some renderings of the games.

    def key(result):
        snake_failed = result.benchmark in SNAKE_BENCHMARKS and not result.passed
        return (
            result.benchmark,
            not result.passed,
            -result.score,
            # Failed Snakes rank the longest survival first.
            _longest_game_first(result) if snake_failed else result.parameter_count,
            result.parameter_count if snake_failed else _game_length(result),
            _mean_error(result),
            result.submission,
        )

    return sorted(results, key=key)


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
