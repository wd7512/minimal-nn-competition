"""Pendulum benchmark: a 5-logit net controls a discrete-torque pendulum."""

import random
import statistics

from minimal_nn_competition.benchmarks.pendulum.engine import observation, reset, step
from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_HORIZON = 200
INPUT_SIZE = 3
OUTPUT_SIZE = 5
# Tuned after measuring the PPO example and a uniform random policy; see
# `models/pendulum/example` and tests/test_rl_environments.py.
PASS_THRESHOLD = -500.0


class PendulumBenchmark:
    name = "pendulum"

    def __init__(self, seeds=DEFAULT_SEEDS, horizon=DEFAULT_HORIZON, threshold=PASS_THRESHOLD):
        if not seeds or horizon <= 0:
            raise ValueError("Pendulum benchmark requires seeds and positive horizon")
        self.seeds = seeds
        self.horizon = horizon
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
                for _ in range(self.horizon):
                    features = observation(state)
                    inputs = torch.tensor(features, dtype=torch.float32)[None]
                    action = int(submission.model(inputs).argmax(dim=1).item())
                    state, reward, _ = step(state, action)
                    total += reward
                returns.append(total)
        score = statistics.fmean(returns)
        return (
            score,
            statistics.pstdev(returns),
            score >= self.threshold,
            {
                "seed_returns": returns,
                "horizon": self.horizon,
                "seeds": len(self.seeds),
                "threshold": self.threshold,
            },
        )
