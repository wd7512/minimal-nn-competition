"""Train a minimal linear connect-4 submission and write it to disk."""

import argparse
import json
import logging
from pathlib import Path

INPUT_SIZE = 42
OUTPUT_SIZE = 7


def create_model(seed=0):
    """Return the linear policy shape used by the example submission."""
    import torch

    torch.manual_seed(seed)
    return torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)


def train(output_dir, seed=0, name=None):
    """Write ``config.json`` and ``model.pt`` for a linear connect-4 net."""
    import torch

    model = create_model(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "connect4",
        "name": name or output_dir.name,
        "author": "example",
        "architecture": "linear",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
    }
    (output_dir / "config.json").write_text(json.dumps(config))
    torch.save(model.state_dict(), output_dir / "model.pt")
    return config


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/connect4/example"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    train(args.output, args.seed, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
