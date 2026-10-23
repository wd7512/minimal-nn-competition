"""Shared evaluation for 28x28 image-classification benchmarks."""

from minimal_nn_competition.model.entry import SubmissionError, materialize_submission


class ImageClassificationBenchmark:
    """Evaluate a 784-input, 10-output classifier against a cached test set."""

    name: str
    display_name: str

    def __init__(self, data_path=None, threshold=1.0):
        self.data_path = data_path
        self.threshold = threshold

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
                f"{name} test data is unavailable; provide the official test-set cache with --data-path"
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
            with torch.no_grad():
                predictions = submission.model(inputs).argmax(1)
            accuracy = (predictions == labels).float().mean().item()
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SubmissionError(
                f"{name} cache must contain raw (inputs, labels) tensors, and the checkpoint must match the declared architecture"
            ) from exc
        return (
            accuracy,
            0.0,
            accuracy >= self.threshold,
            {
                "samples": len(labels),
                "normalisation": config.normalisation,
            },
        )
