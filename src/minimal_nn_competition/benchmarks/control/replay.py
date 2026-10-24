"""Record a deterministic control game for static site playback."""

import random
import statistics

from minimal_nn_competition.benchmarks.cartpole import benchmark as cartpole_benchmark
from minimal_nn_competition.benchmarks.cartpole import engine as cartpole_engine
from minimal_nn_competition.benchmarks.lunarlander import benchmark as lunarlander_benchmark
from minimal_nn_competition.benchmarks.lunarlander import engine as lunarlander_engine
from minimal_nn_competition.benchmarks.pendulum import benchmark as pendulum_benchmark
from minimal_nn_competition.benchmarks.pendulum import engine as pendulum_engine
from minimal_nn_competition.benchmarks.snake.replay import (
    REPLAY_SEED,
    load_replay,
    replay_slug,
    replay_stem,
    save_replay,
)

__all__ = [
    "REPLAY_SEED",
    "load_replay",
    "record_game",
    "replay_slug",
    "replay_stem",
    "save_replay",
]


def _frame(state, action, reward):
    return {"state": [float(value) for value in state], "action": action, "reward": reward}


def _record_cartpole(model, seed, max_steps):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CartPole replays") from exc
    terminated = False
    frames = []
    state = cartpole_engine.reset(random.Random(seed))
    frames.append(_frame(state, None, 0.0))
    total = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            observation = torch.tensor(state, dtype=torch.float32)[None]
            action = int(model(observation).argmax(dim=1).item())
            state, reward, done = cartpole_engine.step(state, action)
            total += reward
            frames.append(_frame(state, action, reward))
            if done:
                terminated = True
                break
    return {
        "kind": "cartpole",
        "seed": seed,
        "steps": len(frames) - 1,
        "total_reward": total,
        "terminated": terminated,
        "landed": None,
        "frames": frames,
    }


def _record_pendulum(model, seed, horizon):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Pendulum replays") from exc
    state = pendulum_engine.reset(random.Random(seed))
    frames = [_frame(state, None, 0.0)]
    total = 0.0
    with torch.no_grad():
        for _ in range(horizon):
            features = pendulum_engine.observation(state)
            inputs = torch.tensor(features, dtype=torch.float32)[None]
            action = int(model(inputs).argmax(dim=1).item())
            state, reward, _ = pendulum_engine.step(state, action)
            total += reward
            frames.append(_frame(state, action, reward))
    return {
        "kind": "pendulum",
        "seed": seed,
        "steps": len(frames) - 1,
        "total_reward": total,
        "terminated": False,
        "landed": None,
        "frames": frames,
    }


def _record_lunarlander(model, seed, max_steps):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for LunarLander replays") from exc
    terminated = False
    success = False
    state = lunarlander_engine.reset(random.Random(seed))
    frames = [_frame(state, None, 0.0)]
    total = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            features = lunarlander_engine.observation(state)
            inputs = torch.tensor(features, dtype=torch.float32)[None]
            action = int(model(inputs).argmax(dim=1).item())
            state, reward, done = lunarlander_engine.step(state, action)
            total += reward
            frames.append(_frame(state, action, reward))
            if done:
                terminated = True
                success = state[1] <= 0.0 and lunarlander_engine.landed(state)
                break
    return {
        "kind": "lunarlander",
        "seed": seed,
        "steps": len(frames) - 1,
        "total_reward": total,
        "terminated": terminated,
        "landed": success,
        "frames": frames,
    }


def record_game(benchmark, model, seed=REPLAY_SEED, steps=None):
    """Play one game and return a JSON-serialisable replay payload."""
    if benchmark == "cartpole":
        if steps is None:
            steps = cartpole_benchmark.DEFAULT_MAX_STEPS
        return _record_cartpole(model, seed, steps)
    if benchmark == "pendulum":
        if steps is None:
            steps = pendulum_benchmark.DEFAULT_HORIZON
        return _record_pendulum(model, seed, steps)
    if benchmark == "lunarlander":
        if steps is None:
            steps = lunarlander_benchmark.DEFAULT_MAX_STEPS
        return _record_lunarlander(model, seed, steps)
    raise ValueError(f"Unknown control benchmark: {benchmark}")


def mean_steps(result):
    """Return the mean ± std episode length, mirroring snake's game length."""
    steps = result.details.get("seed_steps")
    if isinstance(steps, list) and steps:
        return f"{statistics.fmean(steps):.1f} ± {statistics.pstdev(steps):.1f}"
    horizon = result.details.get("horizon")
    if isinstance(horizon, int):
        return f"{horizon} ± 0.0"
    return "-"
