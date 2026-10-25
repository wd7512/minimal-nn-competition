"""Hand-rolled Gym-style pendulum dynamics with five discrete torques."""

import math

GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0
MAX_SPEED = 8.0
MAX_TORQUE = 2.0
DT = 0.05
INITIAL_ANGLE_LIMIT = math.pi
INITIAL_ANGULAR_VELOCITY_LIMIT = 2.0
TORQUES = (-MAX_TORQUE, -MAX_TORQUE / 2, 0.0, MAX_TORQUE / 2, MAX_TORQUE)


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
