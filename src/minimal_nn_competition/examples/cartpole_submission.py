"""Train a minimal linear cartpole submission and write it to disk."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

INPUT_SIZE = 4
OUTPUT_SIZE = 2


def train(output_dir: Path, seed: int = 0, name: str | None = None) -> dict[str, Any]:
    """Write ``config.json`` and ``model.pt`` for a linear cartpole net."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "cartpole",
        "name": name or output_dir.name,
        "author": "example",
        "architecture": "linear",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), output_dir / "model.pt")
    return config


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/cartpole/example"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Display name (defaults to output dir)")
    args = parser.parse_args(argv)
    train(args.output, seed=args.seed, name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
