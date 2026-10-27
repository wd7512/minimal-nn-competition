"""Shared evaluation for 28x28 image-classification benchmarks."""

from minimal_nn_competition.model.entry import SubmissionError, materialize_submission


class ImageClassificationBenchmark:
    """Evaluate a 784-input, 10-output classifier against the full dataset."""

    name: str
    display_name: str
    expected_samples: int = 70_000
    batch_size: int = 512

    def __init__(self, data_path=None, threshold=1.0, expected_samples=None, batch_size=512):
        self.data_path = data_path
        self.threshold = threshold
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.expected_samples = (
            type(self).expected_samples if expected_samples is None else expected_samples
        )

    def load_submission(self, submission):
        config, name = submission.config, self.display_name
        if config.benchmark != self.name:
            raise ValueError(f"Submission benchmark does not match {name}")
        if config.input_size != 784 or config.output_size != 10:
            raise ValueError(f"{name} submissions must declare input_size 784 and output_size 10")
        return materialize_submission(submission)

    def evaluate(self, submission):
        config, name = submission.config, self.display_name
        if self.data_path is None or not self.data_path.is_file():
            raise FileNotFoundError(
                f"{name} dataset is unavailable; provide the official dataset cache with --data-path"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"PyTorch is required for {name} evaluation") from exc
        try:
            inputs, labels = torch.load(self.data_path, map_location="cpu", weights_only=True)
            if inputs.ndim != 2 or inputs.shape[1] != 784:
                raise ValueError(f"{name} inputs must be flattened to 784 values")
            inputs = inputs.float()
            if config.normalisation:
                inputs = inputs / 255.0
            labels = labels.flatten()
            if len(inputs) != len(labels):
                raise ValueError(f"{name} inputs and labels must have the same length")
            if len(labels) == 0 or (labels < 0).any().item() or (labels > 9).any().item():
                raise ValueError(f"{name} labels must be non-empty class indices from 0 to 9")
            if self.expected_samples is not None and len(labels) != self.expected_samples:
                raise SubmissionError(
                    f"{name} dataset must contain exactly {self.expected_samples} samples, "
                    f"found {len(labels)}"
                )
            correct = 0
            with torch.no_grad():
                for start in range(0, len(labels), self.batch_size):
                    batch = inputs[start : start + self.batch_size]
                    logits = submission.model(batch)
                    if logits.ndim != 2 or logits.shape != (len(batch), 10):
                        raise SubmissionError(
                            f"{name} model must return one ten-class row per input"
                        )
                    if not torch.isfinite(logits).all().item():
                        raise SubmissionError(f"{name} model returned non-finite scores")
                    predictions = logits.argmax(1)
                    correct += int((predictions == labels[start : start + len(batch)]).sum())
            accuracy = correct / len(labels)
        except SubmissionError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SubmissionError(
                f"{name} cache must contain raw (inputs, labels) tensors, and the checkpoint must match the declared architecture"
            ) from exc
        return (
            accuracy,
            accuracy >= self.threshold,
            {
                "samples": len(labels),
                "normalisation": config.normalisation,
            },
        )
