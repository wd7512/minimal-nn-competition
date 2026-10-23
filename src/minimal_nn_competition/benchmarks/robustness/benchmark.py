"""Robustness benchmark: PGD-Linf accuracy on cached test tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minimal_nn_competition.benchmarks.image_classification import (
    ImageClassificationBenchmark,
)
from minimal_nn_competition.model.entry import LoadedSubmission, SubmissionError


class RobustnessBenchmark(ImageClassificationBenchmark):
    name = "robustness"
    display_name = "Robustness"

    def __init__(
        self,
        data_path: Path | None = None,
        threshold: float = 0.0,
        eps: float = 0.1,
        steps: int = 10,
        samples: int = 1024,
    ) -> None:
        super().__init__(data_path=data_path, threshold=threshold)
        self.eps = eps
        self.steps = steps
        self.samples = samples

    def _pgd(self, model: Any, inputs: Any, labels: Any, scale: float) -> Any:
        import torch

        step = self.eps / max(self.steps, 1)
        adv = inputs.clone()
        for _ in range(self.steps):
            current = adv.detach().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(model(current * scale), labels)
            grad = torch.autograd.grad(loss, current)[0]
            adv = (adv + step * grad.sign()).clamp(0, 1)
            adv = torch.max(torch.min(adv, inputs + self.eps), inputs - self.eps).clamp(0, 1)
        return adv

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
            base = inputs.float()[: self.samples] / 255.0
            labels = labels.flatten()[: self.samples]
            if base.shape[0] != labels.shape[0]:
                raise ValueError(f"{self.display_name} inputs and labels must have the same length")
            scale = 1.0 if submission.config.normalisation else 255.0
            with torch.no_grad():
                clean = torch.argmax(submission.model(base * scale), dim=1)
            clean_accuracy = float((clean == labels).float().mean().item())
            adv = self._pgd(submission.model, base, labels, scale)
            with torch.no_grad():
                robust = torch.argmax(submission.model(adv * scale), dim=1)
            robust_accuracy = float((robust == labels).float().mean().item())
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SubmissionError(
                f"{self.display_name} cache must contain raw (inputs, labels) tensors, and the "
                "checkpoint must match the declared architecture"
            ) from exc
        return (
            robust_accuracy,
            0.0,
            robust_accuracy >= self.threshold,
            {
                "eps": self.eps,
                "steps": self.steps,
                "samples": len(labels),
                "clean_accuracy": clean_accuracy,
                "normalisation": submission.config.normalisation,
            },
        )
