"""Benchmark execution and result serialization."""

import json
import subprocess
from dataclasses import asdict, dataclass
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
    """Return the date the submission config was first committed, if available."""
    config_path = submission.checkpoint.parent / "config.json"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(config_path.parent),
                "log",
                "--follow",
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
