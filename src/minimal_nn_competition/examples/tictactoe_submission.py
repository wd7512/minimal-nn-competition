"""Train a minimal linear tic-tac-toe submission and write it to disk."""

import argparse
import json
from pathlib import Path

INPUT_SIZE = 9
OUTPUT_SIZE = 9


def create_model(seed=0):
    """Return the linear policy shape used by the example submission."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Linear(INPUT_SIZE, OUTPUT_SIZE)


def train(output_dir, seed=0, name=None):
    """Write ``config.json`` and ``model.pt`` for a linear tic-tac-toe net."""
    import torch

    model = create_model(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "tictactoe",
        "name": name or output_dir.name,
        "author": "example",
        "architecture": "linear",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), output_dir / "model.pt")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/tictactoe/example"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Display name (defaults to output dir)")
    args = parser.parse_args(argv)
    train(args.output, seed=args.seed, name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
