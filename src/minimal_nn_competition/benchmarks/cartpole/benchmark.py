"""CartPole benchmark: a 2-logit net balances the pole for max_steps per seed."""

import random
import statistics

import torch

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
        config = submission.config
        if config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match CartPole")
        if config.input_size != INPUT_SIZE:
            raise ValueError("CartPole submissions must declare input_size 4")
        if config.output_size != OUTPUT_SIZE:
            raise ValueError("CartPole submissions must declare output_size 2")
        return materialize_submission(submission)

    def evaluate(self, submission):
        counts = []
        with torch.no_grad():
            for seed in self.seeds:
                state = reset(random.Random(seed))
                balanced = 0
                for _ in range(self.max_steps):
                    observation = torch.tensor(state)
                    action = int(torch.argmax(submission.model(observation[None]), dim=1))
                    state, done = step(state, action)
                    balanced += 1
                    if done:
                        break
                counts.append(balanced)
        return (
            statistics.fmean(counts),
            statistics.pstdev(counts),
            min(counts) >= self.max_steps,
            {"steps": counts, "max_steps": self.max_steps, "seeds": len(self.seeds)},
        )
