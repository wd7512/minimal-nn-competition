"""Benchmark execution and result serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from minimal_nn_competition.model.entry import LoadedSubmission, Submission, parameter_count


@dataclass(frozen=True)
class BenchmarkResult:
    submission: str
    benchmark: str
    score: float
    passed: bool
    parameter_count: int
    details: dict[str, Any]
    score_std: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def evaluate_submission(submission: Submission, benchmark: Any) -> BenchmarkResult:
    loaded: LoadedSubmission = benchmark.load_submission(submission)
    score, score_std, passed, details = benchmark.evaluate(loaded)
    return BenchmarkResult(
        submission=submission.checkpoint.parent.name,
        benchmark=benchmark.name,
        score=float(score),
        passed=bool(passed),
        parameter_count=parameter_count(loaded.state_dict),
        details=dict(details),
        score_std=float(score_std),
    )
