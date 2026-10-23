"""Train a minimal MLP parity submission and write it to disk."""

import argparse
import json
import logging
from pathlib import Path

from minimal_nn_competition.benchmarks.parity.benchmark import sample_dataset

logger = logging.getLogger(__name__)

N_BITS = 16
OUTPUT_SIZE = 2
HIDDEN_SIZE = 128


def create_model(n_bits=N_BITS, seed=0):
    """Return the MLP shape used by the example submission."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_bits, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    )


def train(output_dir, n_bits=N_BITS, steps=200, seed=0, name=None):
    """Train briefly on sampled parity data, then write config and model."""
    import torch

    model = create_model(n_bits, seed)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for step in range(steps):
        inputs, labels = sample_dataset(n_bits, seed + step, 512)
        batch = torch.tensor(inputs, dtype=torch.float32)
        expected = torch.tensor(labels, dtype=torch.long)
        loss = loss_fn(model(batch), expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "parity",
        "name": name or output_dir.name,
        "author": "example",
        "architecture": "mlp",
        "input_size": n_bits,
        "output_size": OUTPUT_SIZE,
        "hidden_sizes": [HIDDEN_SIZE],
    }
    (output_dir / "config.json").write_text(json.dumps(config))
    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info("Wrote parity example submission to %s", output_dir)
    return config


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/parity/example"))
    parser.add_argument("--n-bits", type=int, default=N_BITS)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Display name (defaults to output dir)")
    args = parser.parse_args(argv)
    train(args.output, args.n_bits, args.steps, args.seed, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
