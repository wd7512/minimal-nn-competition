"""Record a deterministic control game for static site playback."""

import random
import statistics

from minimal_nn_competition.benchmarks import cartpole, lunarlander, pendulum
from minimal_nn_competition.benchmarks.common.snake.replay import (
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
    state = cartpole.reset(random.Random(seed))
    frames.append(_frame(state, None, 0.0))
    total = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            observation = torch.tensor(state, dtype=torch.float32)[None]
            action = int(model(observation).argmax(dim=1).item())
            state, reward, done = cartpole.step(state, action)
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
    state = pendulum.reset(random.Random(seed))
    frames = [_frame(state, None, 0.0)]
    total = 0.0
    with torch.no_grad():
        for _ in range(horizon):
            features = pendulum.observation(state)
            inputs = torch.tensor(features, dtype=torch.float32)[None]
            action = int(model(inputs).argmax(dim=1).item())
            state, reward, _ = pendulum.step(state, action)
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
    state = lunarlander.reset(random.Random(seed))
    frames = [_frame(state, None, 0.0)]
    total = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            features = lunarlander.observation(state)
            inputs = torch.tensor(features, dtype=torch.float32)[None]
            action = int(model(inputs).argmax(dim=1).item())
            state, reward, done = lunarlander.step(state, action)
            total += reward
            frames.append(_frame(state, action, reward))
            if done:
                terminated = True
                success = state[1] <= 0.0 and lunarlander.landed(state)
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
            steps = cartpole.DEFAULT_MAX_STEPS
        return _record_cartpole(model, seed, steps)
    if benchmark == "pendulum":
        if steps is None:
            steps = pendulum.DEFAULT_MAX_STEPS
        return _record_pendulum(model, seed, steps)
    if benchmark == "lunarlander":
        if steps is None:
            steps = lunarlander.DEFAULT_MAX_STEPS
        return _record_lunarlander(model, seed, steps)
    raise ValueError(f"Unknown control benchmark: {benchmark}")


def steps_text(result):
    """Return the mean ± std episode steps, mirroring snake's game length."""
    steps = result.details.get("seed_steps")
    if isinstance(steps, list) and steps:
        return f"{statistics.fmean(steps):.1f} ± {statistics.pstdev(steps):.1f}"
    return "-"
