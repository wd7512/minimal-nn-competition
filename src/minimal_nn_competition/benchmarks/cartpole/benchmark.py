"""CartPole benchmark: a 2-logit net balances the pole for max_steps per seed."""

from __future__ import annotations

import random
import statistics
from typing import Any

from minimal_nn_competition.benchmarks.cartpole.engine import reset, step
from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    materialize_submission,
)

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 500
INPUT_SIZE = 4
OUTPUT_SIZE = 2


class CartPoleBenchmark:
    name = "cartpole"

    def __init__(
        self,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if not seeds:
            raise ValueError("CartPole benchmark requires at least one seed")
        if max_steps <= 0:
            raise ValueError("CartPole benchmark max_steps must be positive")
        self.seeds = seeds
        self.max_steps = max_steps

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match CartPole")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("CartPole submissions must declare input_size 4")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("CartPole submissions must declare output_size 2")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CartPole evaluation") from exc
        counts = []
        with torch.no_grad():
            for seed in self.seeds:
                state = reset(random.Random(seed))
                balanced = 0
                for _ in range(self.max_steps):
                    observation = torch.tensor(state, dtype=torch.float32)
                    action = int(
                        torch.argmax(submission.model(observation.unsqueeze(0)), dim=1).item()
                    )
                    state, done = step(state, action)
                    balanced += 1
                    if done:
                        break
                counts.append(balanced)
        return (
            statistics.fmean(counts),
            statistics.pstdev(counts),
            all(count == self.max_steps for count in counts),
            {"steps": counts, "max_steps": self.max_steps, "seeds": len(self.seeds)},
        )
