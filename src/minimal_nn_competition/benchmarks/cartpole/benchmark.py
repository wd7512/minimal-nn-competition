"""CartPole benchmark: a 2-logit net balances the pole for max_steps per seed.

Score is the fraction of seeds that survive the full horizon (1.0 = every seed
passes); mean steps stay in details as the reward column.
"""

import random
import statistics

from minimal_nn_competition.benchmarks.cartpole.engine import reset, step
from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 500
INPUT_SIZE = 4
OUTPUT_SIZE = 2


class CartPoleBenchmark:
    name = "cartpole"

    def __init__(self, seeds=DEFAULT_SEEDS, max_steps=DEFAULT_MAX_STEPS):
        if not seeds or max_steps <= 0:
            raise ValueError("CartPole benchmark requires seeds and positive max_steps")
        self.seeds = seeds
        self.max_steps = max_steps

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match CartPole")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("CartPole submissions must declare input_size 4")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("CartPole submissions must declare output_size 2")
        return materialize_submission(submission)

    def evaluate(self, submission):
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
                    observation = torch.tensor(state, dtype=torch.float32)[None]
                    action = int(submission.model(observation).argmax(dim=1).item())
                    state, _, done = step(state, action)
                    balanced += 1
                    if done:
                        break
                counts.append(balanced)
        passed_flags = [count == self.max_steps for count in counts]
        return (
            statistics.fmean(passed_flags),
            all(passed_flags),
            {
                "seed_steps": counts,
                "mean_steps": statistics.fmean(counts),
                "max_steps": self.max_steps,
                "seeds": len(self.seeds),
            },
        )
