"""Load, save, rank, and update the self-applied speed leaderboard."""

import json
from dataclasses import asdict

from minimal_nn_competition.speed.benchmark import SpeedResult


def rank_speed_results(results):
    return sorted(results, key=lambda result: (-result.ticks_per_second, result.commit))


def save_speed_results(path, results):
    payload = [asdict(result) for result in rank_speed_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_speed_results(path):
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Speed leaderboard must contain a JSON list")
    results = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Speed leaderboard entries must be JSON objects")
        results.append(_load_speed_result(item))
    return rank_speed_results(results)


def _load_speed_result(item):
    run_ticks = [int(value) for value in item.get("run_ticks", [])]
    return SpeedResult(
        commit=str(item.get("commit", "unknown")),
        ticks_per_second=float(item["ticks_per_second"]),
        ticks_std=float(item.get("ticks_std", 0.0)),
        run_ticks=run_ticks,
        measured_at=str(item.get("measured_at", "")),
        duration_seconds=float(item.get("duration_seconds", 1.0)),
        runs=int(item.get("runs", len(run_ticks))),
        warmup_seconds=float(item.get("warmup_seconds", 0.0)),
        agent=str(item.get("agent", "unknown")),
        harness=str(item.get("harness", "unknown")),
        python_version=str(item.get("python_version", "")),
        platform=str(item.get("platform", "")),
        machine=str(item.get("machine", "")),
        cpu=str(item.get("cpu", "")),
        validation_seeds=int(item.get("validation_seeds", 0)),
    )


def upsert_speed_result(results, result):
    return [existing for existing in results if existing.commit != result.commit] + [result]
