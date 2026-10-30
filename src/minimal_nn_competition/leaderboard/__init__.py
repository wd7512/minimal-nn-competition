"""A github pages leaderboard for the minimal neural network competition."""

import json
import statistics
from dataclasses import asdict
from datetime import date

from minimal_nn_competition.benchmarks.registry import CHESS_BENCHMARKS, SNAKE_BENCHMARKS
from minimal_nn_competition.runner.entry import BenchmarkResult, validate_result


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


def _submission_date(result):
    """Older submissions rank first; invalid dates are rejected at IO boundaries."""
    try:
        return date.fromisoformat(result.submitted_at)
    except (AttributeError, TypeError, ValueError):
        return date.max


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
            _submission_date(result),
            result.name.casefold() if isinstance(result.name, str) else "",
            result.submission,
        )

    return sorted(results, key=key)


def save_results(path, results):
    results = list(results)
    for index, result in enumerate(results, 1):
        validate_result(result, context=f"leaderboard result {index}")
    payload = [asdict(result) for result in rank_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_results(path):
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Leaderboard must contain a JSON list")
    results = []
    required_fields = {
        "submission",
        "benchmark",
        "score",
        "passed",
        "parameter_count",
        "details",
        "name",
        "author",
        "submitted_at",
    }
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Leaderboard entry {index} must be a JSON object")
        missing = sorted(required_fields - item.keys())
        if missing:
            raise ValueError(
                f"Leaderboard entry {index} is missing required fields: {', '.join(missing)}"
            )
        result = BenchmarkResult(
            submission=item["submission"],
            benchmark=item["benchmark"],
            score=item["score"],
            passed=item["passed"],
            parameter_count=item["parameter_count"],
            details=item["details"],
            name=item["name"],
            author=item["author"],
            submitted_at=item["submitted_at"],
        )
        validate_result(result, context=f"leaderboard entry {index}")
        results.append(result)
    return results
