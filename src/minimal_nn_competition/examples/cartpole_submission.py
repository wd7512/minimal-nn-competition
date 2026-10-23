"""Train a minimal linear cartpole submission and write it to disk."""

import argparse
import json
from pathlib import Path

INPUT_SIZE = 4
OUTPUT_SIZE = 2


def train(output_dir, seed=0, name=None):
    """Write ``config.json`` and ``model.pt`` for a linear cartpole net."""
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "cartpole",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/cartpole/example"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="Display name (defaults to output dir)")
    args = parser.parse_args(argv)
    train(args.output, args.seed, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
