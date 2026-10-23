"""Hand-rolled CartPole dynamics (Sutton-Barto constants, no dependencies)."""

from __future__ import annotations

import math
import random

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * 2.0 * math.pi / 360.0

State = tuple[float, float, float, float]


def reset(rng: random.Random) -> State:
    """Return a near-upright state with small uniform noise."""
    return (
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
    )


def step(state: State, action: int) -> tuple[State, bool]:
    """Advance semi-implicit Euler dynamics; return (next state, done)."""
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    total_mass = MASSCART + MASSPOLE
    polemass_length = MASSPOLE * LENGTH
    temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta * cos_theta / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass
    x_dot += TAU * x_acc
    x += TAU * x_dot
    theta_dot += TAU * theta_acc
    theta += TAU * theta_dot
    done = abs(x) > X_THRESHOLD or abs(theta) > THETA_THRESHOLD
    return (x, x_dot, theta, theta_dot), done
