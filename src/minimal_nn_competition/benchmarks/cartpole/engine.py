"""Hand-rolled CartPole dynamics (Sutton-Barto constants, no dependencies)."""

import math

MASSPOLE = 0.1
LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02
X_LIMIT = 2.4
THETA_LIMIT = 0.20944


def reset(rng):
    """Return a near-upright state with small uniform noise."""
    return (
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
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
