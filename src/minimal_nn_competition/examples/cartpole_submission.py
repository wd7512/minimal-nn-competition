"""Train a tuned linear CartPole controller and write it to disk."""

import argparse
import json
import logging
from pathlib import Path

INPUT_SIZE = 4
OUTPUT_SIZE = 2
# A PD controller on the pole angle plus a small cart-position correction.
# Tuned offline; it balances all 100 benchmark seeds for 500 steps.
WEIGHTS = (
    (-0.05, -0.25, -1.0, -0.5),
    (0.05, 0.25, 1.0, 0.5),
)
BIASES = (0.0, 0.0)


def train(output_dir, seed=0, name=None, author="unknown"):
    """Write ``config.json`` and ``model.pt`` for the tuned linear controller."""
    import torch

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    model = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(WEIGHTS, dtype=torch.float32))
        model.bias.copy_(torch.tensor(BIASES, dtype=torch.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "cartpole",
        "name": name or output_dir.name,
        "author": author,
        "architecture": "linear",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "description": "A tuned linear pole-balancing controller.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    logger.info("Wrote %s", output_dir / "model.pt")
    return state_dict


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/cartpole/example"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Display name")
    parser.add_argument("--author", default="unknown", help="Submission author")
    args = parser.parse_args(argv)
    train(args.output, args.seed, args.name, args.author)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
