"""CartPole benchmark: a 2-logit net balances the pole for max_steps per seed.

Score is the fraction of seeds that survive the full horizon (1.0 = every seed
passes); mean steps stay in details as the reward column. The hand-rolled
dynamics below use the Sutton-Barto constants and need no extra dependencies.
"""

import math
import random
import statistics

from minimal_nn_competition.model.entry import materialize_submission

__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SEEDS",
    "INPUT_SIZE",
    "OUTPUT_SIZE",
    "CartPoleBenchmark",
    "reset",
    "step",
]

MASSPOLE = 0.1
LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02
X_LIMIT = 2.4
THETA_LIMIT = math.radians(40.0)
INITIAL_X_LIMIT = 1.5
INITIAL_THETA_LIMIT = math.radians(25.0)
INITIAL_VELOCITY_LIMIT = 0.02
INITIAL_ANGULAR_VELOCITY_LIMIT = 0.02

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 500
INPUT_SIZE = 4
OUTPUT_SIZE = 2


def reset(rng):
    """Return a varied state with bounded uniform initialization noise."""
    return (
        rng.uniform(-INITIAL_X_LIMIT, INITIAL_X_LIMIT),
        rng.uniform(-INITIAL_VELOCITY_LIMIT, INITIAL_VELOCITY_LIMIT),
        rng.uniform(-INITIAL_THETA_LIMIT, INITIAL_THETA_LIMIT),
        rng.uniform(-INITIAL_ANGULAR_VELOCITY_LIMIT, INITIAL_ANGULAR_VELOCITY_LIMIT),
    )


def step(state, action):
    """Advance semi-implicit Euler dynamics; return (next state, reward, done)."""
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action else -FORCE_MAG
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    total_mass = 1.0 + MASSPOLE
    polemass_length = MASSPOLE * LENGTH
    temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass
    theta_acc = (9.8 * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta * cos_theta / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass
    x_dot += TAU * x_acc
    x += TAU * x_dot
    theta_dot += TAU * theta_acc
    theta += TAU * theta_dot
    done = abs(x) > X_LIMIT or abs(theta) > THETA_LIMIT
    return (x, x_dot, theta, theta_dot), 0.0 if done else 1.0, done


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
            {"seed_steps": counts},
        )
