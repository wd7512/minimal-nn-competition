"""Record a deterministic Snake game for static site playback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

REPLAY_SEED = 0

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def replay_slug(submission: str) -> str:
    """Map a submission name to a safe file stem."""
    return _SLUG_RE.sub("_", submission).strip("_") or "game"


def record_game(model: Any, seed: int = REPLAY_SEED, max_moves: int = 4096) -> dict[str, Any]:
    """Play one game and return a JSON-serialisable replay payload."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Snake replays") from exc
    environment = SnakeEnvironment(seed=seed, max_moves=max_moves)
    observation = environment.reset()
    frames = [{"snake": [list(cell) for cell in environment.snake], "food": list(environment.food)}]
    done = False
    with torch.no_grad():
        while not done:
            values = model(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(values, dim=1).item())
            observation, _, done = environment.step(action)
            frames.append(
                {
                    "snake": [list(cell) for cell in environment.snake],
                    "food": list(environment.food),
                }
            )
    return {
        "seed": seed,
        "size": environment.size,
        "length": len(environment.snake),
        "moves": len(frames) - 1,
        "frames": frames,
    }


def save_replay(path: Path, payload: dict[str, Any]) -> Path:
    """Write a replay payload to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def load_replay(path: Path) -> dict[str, Any]:
    """Read a replay payload written by :func:`save_replay`."""
    raw: Any = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
        raise ValueError(f"Invalid replay file: {path}")
    return raw
