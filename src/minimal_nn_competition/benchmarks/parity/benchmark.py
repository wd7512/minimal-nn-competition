"""N-bit parity benchmark: predict even/odd from N binary inputs."""

from __future__ import annotations

import random
import statistics
from typing import Any

from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    materialize_submission,
)

DEFAULT_SEEDS = tuple(range(100))
SAMPLES_PER_SEED = 1024


def sample_dataset(
    n_bits: int, seed: int, count: int = SAMPLES_PER_SEED
) -> tuple[list[list[float]], list[int]]:
    """Sample a deterministic eval set of bit-strings and parity labels."""
    rng = random.Random(seed * 65537 + n_bits)
    inputs = [[float(rng.getrandbits(1)) for _ in range(n_bits)] for _ in range(count)]
    labels = [int(sum(row)) % 2 for row in inputs]
    return inputs, labels


class ParityBenchmark:
    name = "parity"

    def __init__(self, n_bits: int = 16, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> None:
        if n_bits <= 0:
            raise ValueError("Parity benchmark requires a positive bit count")
        if not seeds:
            raise ValueError("Parity benchmark requires at least one seed")
        self.n_bits = n_bits
        self.seeds = seeds

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Parity")
        if submission.config.input_size != self.n_bits:
            raise ValueError(f"Parity submissions must declare input_size {self.n_bits}")
        if submission.config.output_size != 2:
            raise ValueError("Parity submissions must declare output_size 2")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Parity evaluation") from exc
        accuracies = []
        with torch.no_grad():
            for seed in self.seeds:
                inputs, labels = sample_dataset(self.n_bits, seed)
                batch = torch.tensor(inputs, dtype=torch.float32)
                expected = torch.tensor(labels, dtype=torch.long)
                predicted = torch.argmax(submission.model(batch), dim=1)
                accuracies.append(float((predicted == expected).float().mean().item()))
        return (
            statistics.fmean(accuracies),
            statistics.pstdev(accuracies),
            all(accuracy == 1.0 for accuracy in accuracies),
            {
                "seed_accuracies": accuracies,
                "seeds": len(self.seeds),
                "n_bits": self.n_bits,
                "samples_per_seed": SAMPLES_PER_SEED,
            },
        )
