"""LunarLander benchmark: a 4-logit net lands a point-mass craft on a pad.

Score is the fraction of seeds that land (1.0 = every seed passes); mean
return stays in details as the reward column. The hand-rolled 2D point-mass
dynamics below need no extra dependencies.
"""

import random
import statistics

from minimal_nn_competition.model.entry import materialize_submission

__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SEEDS",
    "INPUT_SIZE",
    "OUTPUT_SIZE",
    "LunarLanderBenchmark",
    "landed",
    "observation",
    "reset",
    "step",
]

GRAVITY = 0.02
MAIN_THRUST = 0.05
SIDE_THRUST = 0.02
ANGULAR_DAMPING = 0.9
PAD_HALF_WIDTH = 0.25
X_LIMIT = 1.0
Y_LIMIT = 1.5
TOUCHDOWN_Y = 0.05
LANDING_MAX_HORIZONTAL_SPEED = 0.2
LANDING_MAX_VERTICAL_SPEED = 0.2
LANDING_MAX_ANGLE = 0.5
LANDING_REWARD = 10.0
CRASH_PENALTY = 10.0
INITIAL_X_LIMIT = 0.4
INITIAL_Y_MIN = 1.0
INITIAL_Y_MAX = 1.3
INITIAL_VELOCITY_LIMIT = 0.1
INITIAL_ANGLE_LIMIT = 0.2
INITIAL_ANGULAR_VELOCITY_LIMIT = 0.03

DEFAULT_SEEDS = tuple(range(100))
DEFAULT_MAX_STEPS = 300
INPUT_SIZE = 8
OUTPUT_SIZE = 4


def reset(rng):
    """Return a lander above the pad with varied initial drift and tilt."""
    return (
        rng.uniform(-INITIAL_X_LIMIT, INITIAL_X_LIMIT),
        rng.uniform(INITIAL_Y_MIN, INITIAL_Y_MAX),
        rng.uniform(-INITIAL_VELOCITY_LIMIT, INITIAL_VELOCITY_LIMIT),
        rng.uniform(-INITIAL_VELOCITY_LIMIT, INITIAL_VELOCITY_LIMIT),
        rng.uniform(-INITIAL_ANGLE_LIMIT, INITIAL_ANGLE_LIMIT),
        rng.uniform(-INITIAL_ANGULAR_VELOCITY_LIMIT, INITIAL_ANGULAR_VELOCITY_LIMIT),
    )


def _contacts(state):
    x, y, _, _, angle, _ = state
    if y > TOUCHDOWN_Y or abs(x) > PAD_HALF_WIDTH:
        return 0.0, 0.0
    return (1.0 if angle >= 0.0 else 0.0), (1.0 if angle <= 0.0 else 0.0)


def observation(state):
    """Return the eight-value (position, velocity, angle, contact) observation."""
    x, y, vx, vy, angle, angular_velocity = state
    left, right = _contacts(state)
    return (x, y, vx, vy, angle, angular_velocity, left, right)


def landed(state):
    x, y, vx, vy, angle, _ = state
    return (
        abs(x) <= PAD_HALF_WIDTH
        and abs(vx) <= LANDING_MAX_HORIZONTAL_SPEED
        and abs(vy) <= LANDING_MAX_VERTICAL_SPEED
        and abs(angle) <= LANDING_MAX_ANGLE
    )


def step(state, action):
    """Advance one timestep; return (next state, reward, done)."""
    if not 0 <= action < 4:
        raise ValueError("LunarLander action must be in 0..3")
    x, y, vx, vy, angle, angular_velocity = state
    if action == 1:
        vy += MAIN_THRUST
    elif action == 2:
        vx -= SIDE_THRUST
        angular_velocity += SIDE_THRUST
    elif action == 3:
        vx += SIDE_THRUST
        angular_velocity -= SIDE_THRUST

    vy -= GRAVITY
    angle += angular_velocity
    angular_velocity *= ANGULAR_DAMPING
    x += vx
    y += vy

    reward = (
        -0.1
        - 0.5 * abs(x)
        - 0.5 * abs(vx)
        - 0.2 * abs(vy)
        - 0.1 * abs(angle)
        - 0.05 * abs(angular_velocity)
        - (0.01 if action == 1 else 0.0)
    )
    next_state = (x, y, vx, vy, angle, angular_velocity)
    done = False
    if y <= 0.0:
        if landed(next_state):
            reward += LANDING_REWARD
        else:
            reward -= CRASH_PENALTY
        done = True
    elif abs(x) > X_LIMIT or y > Y_LIMIT:
        reward -= CRASH_PENALTY
        done = True
    return next_state, reward, done


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
        passed = []
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
                        success = state[1] <= 0.0 and landed(state)
                        break
                returns.append(total)
                passed.append(success)
                steps.append(taken)
        return (
            statistics.fmean(passed),
            all(passed),
            {
                "seed_returns": returns,
                "seed_landed": passed,
                "seed_steps": steps,
                "mean_return": statistics.fmean(returns),
            },
        )
