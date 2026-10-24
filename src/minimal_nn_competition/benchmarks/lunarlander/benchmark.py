"""LunarLander benchmark: a 4-logit net lands a point-mass craft on a pad.

Score is the fraction of seeds that land (1.0 = every seed passes); mean
return stays in details as the reward column.
"""

import random
import statistics

from minimal_nn_competition.benchmarks.lunarlander.engine import landed as is_landed
from minimal_nn_competition.benchmarks.lunarlander.engine import observation, reset, step
from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 300
INPUT_SIZE = 8
OUTPUT_SIZE = 4


class LunarLanderBenchmark:
    name = "lunarlander"

    def __init__(self, seeds=DEFAULT_SEEDS, max_steps=DEFAULT_MAX_STEPS):
        if not seeds or max_steps <= 0:
            raise ValueError("LunarLander benchmark requires seeds and positive max_steps")
        self.seeds = seeds
        self.max_steps = max_steps

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match LunarLander")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("LunarLander submissions must declare input_size 8")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("LunarLander submissions must declare output_size 4")
        return materialize_submission(submission)

    def evaluate(self, submission):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for LunarLander evaluation") from exc
        returns = []
        landed = []
        steps = []
        with torch.no_grad():
            for seed in self.seeds:
                state = reset(random.Random(seed))
                total = 0.0
                done = False
                success = False
                taken = 0
                for _ in range(self.max_steps):
                    features = observation(state)
                    inputs = torch.tensor(features, dtype=torch.float32)[None]
                    action = int(submission.model(inputs).argmax(dim=1).item())
                    state, reward, done = step(state, action)
                    total += reward
                    taken += 1
                    if done:
                        success = state[1] <= 0.0 and is_landed(state)
                        break
                returns.append(total)
                landed.append(success)
                steps.append(taken)
        return (
            statistics.fmean(landed),
            statistics.pstdev(landed),
            all(landed),
            {
                "seed_returns": returns,
                "seed_landed": landed,
                "seed_steps": steps,
                "mean_return": statistics.fmean(returns),
                "max_steps": self.max_steps,
                "seeds": len(self.seeds),
            },
        )
