"""Pendulum benchmark: a 5-logit net controls a discrete-torque pendulum.

Score is the fraction of seeds whose return clears PASS_THRESHOLD
(1.0 = every seed passes); mean return stays in details as the reward column.
The hand-rolled dynamics below follow the Gym-style pendulum with five
discrete torques.
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
    "PASS_THRESHOLD",
    "PendulumBenchmark",
    "angle_normalize",
    "observation",
    "reset",
    "step",
]

GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0
MAX_SPEED = 8.0
MAX_TORQUE = 2.0
DT = 0.05
INITIAL_ANGLE_LIMIT = math.pi
INITIAL_ANGULAR_VELOCITY_LIMIT = 2.0
TORQUES = (-MAX_TORQUE, -MAX_TORQUE / 2, 0.0, MAX_TORQUE / 2, MAX_TORQUE)

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 200
INPUT_SIZE = 3
OUTPUT_SIZE = 5
# Tuned after measuring the PPO example and a uniform random policy; see
# `models/pendulum/example` and tests/test_rl_environments.py.
PASS_THRESHOLD = -500.0


def angle_normalize(theta):
    """Wrap an angle into [-pi, pi]."""
    return ((theta + math.pi) % (2 * math.pi)) - math.pi


def reset(rng):
    """Return a random angle and a challenging initial angular velocity."""
    return rng.uniform(-INITIAL_ANGLE_LIMIT, INITIAL_ANGLE_LIMIT), rng.uniform(
        -INITIAL_ANGULAR_VELOCITY_LIMIT, INITIAL_ANGULAR_VELOCITY_LIMIT
    )


def observation(state):
    """Return the (cos, sin, angular velocity) observation."""
    theta, theta_dot = state
    return math.cos(theta), math.sin(theta), theta_dot


def step(state, action):
    """Advance one timestep; return (next state, reward, done)."""
    if not 0 <= action < len(TORQUES):
        raise ValueError(f"Pendulum action must be in 0..{len(TORQUES) - 1}")
    theta, theta_dot = state
    torque = TORQUES[action]
    theta_acc = (
        -3.0 * GRAVITY / (2.0 * LENGTH) * math.sin(theta + math.pi)
        + 3.0 / (MASS * LENGTH * LENGTH) * torque
    )
    theta_dot = min(MAX_SPEED, max(-MAX_SPEED, theta_dot + theta_acc * DT))
    theta = angle_normalize(theta + theta_dot * DT)
    reward = -(angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * torque**2)
    return (theta, theta_dot), reward, False


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
