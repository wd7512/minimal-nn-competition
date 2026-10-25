"""Benchmark execution and result serialization."""

import json
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
    )
