"""Train an example snake-hard submission by reusing the snake PPO example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minimal_nn_competition.examples import snake_submission as snake_example

HIDDEN_SIZES = snake_example.HIDDEN_SIZES
INPUT_SIZE = snake_example.INPUT_SIZE
OUTPUT_SIZE = snake_example.OUTPUT_SIZE
create_model = snake_example.create_model


def train(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """Train via the snake example, then relabel the config for snake-hard."""
    state_dict = snake_example.train(output_dir, **kwargs)
    config = json.loads((output_dir / "config.json").read_text())
    config["benchmark"] = "snake-hard"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return state_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/snake-hard/example"))
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--author", default="unknown")
    args = parser.parse_args(argv)
    train(
        args.output,
        iterations=args.iterations,
        device_name=args.device,
        name=args.name,
        author=args.author,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
