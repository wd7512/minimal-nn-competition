"""Benchmark execution and result serialization."""

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from minimal_nn_competition.model.entry import parameter_count


@dataclass(frozen=True)
class BenchmarkResult:
    submission: str
    benchmark: str
    score: float
    passed: bool
    parameter_count: int
    details: dict[str, Any]
    name: str = "unknown"
    author: str = "unknown"
    submitted_at: str | None = None

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def _missing_value_paths(value, path):
    if value is None:
        return [f"{path} must be populated"]
    if isinstance(value, str):
        return (
            []
            if value.strip() and value.strip().casefold() != "unknown"
            else [f"{path} must be populated"]
        )
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path} must be finite"]
    if isinstance(value, dict):
        errors = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{path} contains an empty field name")
            else:
                errors.extend(_missing_value_paths(item, f"{path}.{key}"))
        return errors
    if isinstance(value, (list, tuple)):
        errors = []
        for index, item in enumerate(value):
            errors.extend(_missing_value_paths(item, f"{path}[{index}]"))
        return errors
    return []


def validate_result(result, *, context="benchmark result"):
    """Reject incomplete or malformed records before they enter the leaderboard."""
    errors = []
    for field in ("submission", "benchmark", "name", "author"):
        value = getattr(result, field, None)
        if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "unknown":
            errors.append(f"{field} must be a non-empty string")

    score = getattr(result, "score", None)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        errors.append("score must be a finite number")
    if type(getattr(result, "passed", None)) is not bool:
        errors.append("passed must be a boolean")
    parameter_count = getattr(result, "parameter_count", None)
    if type(parameter_count) is not int or parameter_count <= 0:
        errors.append("parameter_count must be a positive integer")

    details = getattr(result, "details", None)
    if not isinstance(details, dict):
        errors.append("details must be an object")
    elif not details:
        errors.append("details must not be empty")
    else:
        errors.extend(_missing_value_paths(details, "details"))

    submitted_at = getattr(result, "submitted_at", None)
    if not isinstance(submitted_at, str):
        errors.append("submitted_at must be a valid YYYY-MM-DD date")
    else:
        try:
            parsed_date = date.fromisoformat(submitted_at)
        except ValueError:
            errors.append("submitted_at must be a valid YYYY-MM-DD date")
        else:
            if parsed_date.isoformat() != submitted_at:
                errors.append("submitted_at must use YYYY-MM-DD format")

    if errors:
        raise ValueError(f"Invalid {context}: " + "; ".join(errors))


def evaluate_submission(submission, benchmark):
    loaded = benchmark.load_submission(submission)
    score, passed, details = benchmark.evaluate(loaded)
    config = submission.config
    details = dict(details)
    details["structure"] = [config.input_size, *config.hidden_sizes, config.output_size]
    return BenchmarkResult(
        submission=submission.checkpoint.parent.name,
        benchmark=benchmark.name,
        score=float(score),
        passed=bool(passed),
        parameter_count=parameter_count(loaded.state_dict),
        details=details,
        name=submission.config.name,
        author=submission.config.author,
        submitted_at=_submission_date(submission),
    )


def _submission_date(submission):
    """Return the date this submission's config was added, if available."""
    config_path = submission.checkpoint.parent / "config.json"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(config_path.parent),
                "log",
                "--diff-filter=A",
                "-1",
                "--format=%cs",
                "--",
                config_path.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None
