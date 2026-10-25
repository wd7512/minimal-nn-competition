"""Record a deterministic Snake game for static site playback."""

import json
import re

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

REPLAY_SEED = 0

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def replay_slug(text):
    """Map a submission or benchmark name to a safe file stem."""
    return _SLUG_RE.sub("_", text).strip("_") or "game"


def replay_stem(benchmark, submission):
    """Return a replay file stem that is unique across Snake benchmarks."""
    return f"{replay_slug(benchmark)}--{replay_slug(submission)}"


def record_game(
    model,
    seed=REPLAY_SEED,
    max_moves=4096,
    input_indices=None,
    size=8,
    blocked=(),
    loop_on_full=False,
):
    """Play one game and return a JSON-serialisable replay payload."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Snake replays") from exc
    environment = SnakeEnvironment(
        size=size,
        seed=seed,
        max_moves=max_moves,
        input_indices=input_indices,
        blocked=blocked,
        loop_on_full=loop_on_full,
    )

    def frame():
        food = environment.food
        return {
            "snake": [list(cell) for cell in environment.snake],
            "food": None if food is None else list(food),
        }

    observation = environment.reset()
    frames = [frame()]
    done = False
    with torch.no_grad():
        while not done:
            action = model(torch.tensor(observation, dtype=torch.float32)[None]).argmax(1).item()
            observation, _, done = environment.step(action)
            frames.append(frame())
    return {
        "seed": seed,
        "size": environment.size,
        "blocked": [list(cell) for cell in environment.blocked],
        "length": len(environment.snake),
        "moves": len(frames) - 1,
        "frames": frames,
    }


def save_replay(path, payload):
    """Write a replay payload to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def load_replay(path):
    """Read a replay payload written by :func:`save_replay`."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
        raise ValueError(f"Invalid replay file: {path}")
    return raw
