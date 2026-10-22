"""Shared evaluation for 28x28 image-classification benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    SubmissionError,
    materialize_submission,
)


class ImageClassificationBenchmark:
    """Evaluate a 784-input, 10-output classifier against a cached test set."""

    name: str
    display_name: str

    def __init__(self, data_path: Path | None = None, threshold: float = 1.0) -> None:
        self.data_path = data_path
        self.threshold = threshold

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError(f"Submission benchmark does not match {self.display_name}")
        if submission.config.input_size != 784 or submission.config.output_size != 10:
            raise ValueError(
                f"{self.display_name} submissions must declare input_size 784 and output_size 10"
            )
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        if self.data_path is None or not self.data_path.is_file():
            raise FileNotFoundError(
                f"{self.display_name} test data is unavailable; provide the official "
                "test-set cache with --data-path"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"PyTorch is required for {self.display_name} evaluation") from exc
        try:
            data = torch.load(self.data_path, map_location="cpu", weights_only=True)
            inputs, labels = data
            if inputs.ndim != 2 or inputs.shape[1] != 784:
                raise ValueError(f"{self.display_name} inputs must be flattened to 784 values")
            inputs = inputs.float()
            if submission.config.normalisation:
                inputs = inputs / 255.0
            labels = labels.flatten()
            if inputs.shape[0] != labels.shape[0]:
                raise ValueError(f"{self.display_name} inputs and labels must have the same length")
            with torch.no_grad():
                predictions = torch.argmax(submission.model(inputs), dim=1)
            accuracy = float((predictions == labels).float().mean().item())
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SubmissionError(
                f"{self.display_name} cache must contain raw (inputs, labels) tensors, and the "
                "checkpoint must match the declared architecture"
            ) from exc
        return (
            accuracy,
            0.0,
            accuracy >= self.threshold,
            {
                "samples": len(labels),
                "normalisation": submission.config.normalisation,
            },
        )
