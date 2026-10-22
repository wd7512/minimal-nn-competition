"""Load the benchmark ABC into our own benchmark with a the pass/fail criteria and score on minimum parameters"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    SubmissionError,
    materialize_submission,
)

# we need to pick a minimum validation threshold, but i think this is tricky given just the model to avoid overfitting
# So this reframes to a compression challenge where the model should be able to classify all samples across the whole mnist dataset correctly
# scores = {% of correct samples, network_size = number of parameters in the model} ranked by % first then network size


class MnistBenchmark:
    name = "mnist"

    def __init__(self, data_path: Path | None = None, threshold: float = 1.0) -> None:
        self.data_path = data_path
        self.threshold = threshold

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match MNIST")
        if submission.config.input_size != 784 or submission.config.output_size != 10:
            raise ValueError("MNIST submissions must declare input_size 784 and output_size 10")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        if self.data_path is None or not self.data_path.is_file():
            raise FileNotFoundError(
                "MNIST test data is unavailable; provide the official test-set cache "
                "with --data-path"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for MNIST evaluation") from exc
        try:
            data = torch.load(self.data_path, map_location="cpu", weights_only=True)
            inputs, labels = data
            if inputs.ndim != 2 or inputs.shape[1] != 784:
                raise ValueError("MNIST inputs must be flattened to 784 values")
            inputs = inputs.float()
            if submission.config.normalisation:
                inputs = inputs / 255.0
            labels = labels.flatten()
            if inputs.shape[0] != labels.shape[0]:
                raise ValueError("MNIST inputs and labels must have the same length")
            with torch.no_grad():
                predictions = torch.argmax(submission.model(inputs), dim=1)
            accuracy = float((predictions == labels).float().mean().item())
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SubmissionError(
                "MNIST cache must contain raw (inputs, labels) tensors, and the "
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
