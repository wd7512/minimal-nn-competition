"""N-bit parity benchmark: predict even/odd from N binary inputs."""

import random
import statistics

from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(100))
SAMPLES_PER_SEED = 1024


def sample_dataset(n_bits, seed, count=SAMPLES_PER_SEED):
    """Sample a deterministic eval set of bit-strings and parity labels."""
    rng = random.Random(seed + n_bits)
    inputs = [rng.choices((0.0, 1.0), k=n_bits) for _ in range(count)]
    labels = [int(sum(row)) % 2 for row in inputs]
    return inputs, labels


class ParityBenchmark:
    name = "parity"

    def __init__(self, n_bits=16, seeds=DEFAULT_SEEDS):
        if n_bits <= 0 or not seeds:
            raise ValueError("Parity benchmark requires a positive bit count and seeds")
        self.n_bits = n_bits
        self.seeds = seeds

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Parity")
        if submission.config.input_size != self.n_bits or submission.config.output_size != 2:
            raise ValueError(f"Parity submissions must declare input_size {self.n_bits}")
        return materialize_submission(submission)

    def evaluate(self, submission):
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
                predicted = torch.argmax(submission.model(batch), 1)
                accuracies.append((predicted == expected).float().mean().item())
        return (
            statistics.fmean(accuracies),
            statistics.pstdev(accuracies),
            min(accuracies) == 1.0,
            {
                "seed_accuracies": accuracies,
                "seeds": len(self.seeds),
                "n_bits": self.n_bits,
                "samples_per_seed": SAMPLES_PER_SEED,
            },
        )
