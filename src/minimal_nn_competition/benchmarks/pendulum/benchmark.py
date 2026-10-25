"""Pendulum benchmark: a 5-logit net controls a discrete-torque pendulum.

Score is the fraction of seeds whose return clears PASS_THRESHOLD
(1.0 = every seed passes); mean return stays in details as the reward column.
"""

import random
import statistics

from minimal_nn_competition.benchmarks.pendulum.engine import observation, reset, step
from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 200
INPUT_SIZE = 3
OUTPUT_SIZE = 5
# Tuned after measuring the PPO example and a uniform random policy; see
# `models/pendulum/example` and tests/test_rl_environments.py.
PASS_THRESHOLD = -500.0


class PendulumBenchmark:
    name = "pendulum"

    def __init__(self, seeds=DEFAULT_SEEDS, max_steps=DEFAULT_MAX_STEPS, threshold=PASS_THRESHOLD):
        if not seeds or max_steps <= 0:
            raise ValueError("Pendulum benchmark requires seeds and positive max_steps")
        self.seeds = seeds
        self.max_steps = max_steps
        self.threshold = threshold

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Pendulum")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("Pendulum submissions must declare input_size 3")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("Pendulum submissions must declare output_size 5")
        return materialize_submission(submission)

    def evaluate(self, submission):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Pendulum evaluation") from exc
        returns = []
        with torch.no_grad():
            for seed in self.seeds:
                state = reset(random.Random(seed))
                total = 0.0
                for _ in range(self.max_steps):
                    features = observation(state)
                    inputs = torch.tensor(features, dtype=torch.float32)[None]
                    action = int(submission.model(inputs).argmax(dim=1).item())
                    state, reward, _ = step(state, action)
                    total += reward
                returns.append(total)
        passed_flags = [total >= self.threshold for total in returns]
        return (
            statistics.fmean(passed_flags),
            all(passed_flags),
            {"seed_returns": returns, "mean_return": statistics.fmean(returns)},
        )
