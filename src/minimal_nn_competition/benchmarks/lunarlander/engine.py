"""Hand-rolled 2D point-mass lunar lander with four discrete actions."""

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
