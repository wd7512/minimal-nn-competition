"""load the eval function for the mnist benchmark
and train a NN that passes the benchmark with supervised learning"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

HIDDEN_SIZES = (64,)


def create_model() -> Any:
    """Return the MLP shape used by the example submission."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(784, HIDDEN_SIZES[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[0], 10),
    )


def _select_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float | None = None,
    device_name: str = "auto",
    seed: int = 0,
    normalise: bool = False,
    name: str | None = None,
    author: str = "unknown",
) -> dict[str, Any]:
    """Train the example MLP and write ``config.json`` and ``model.pt``."""
    import torch

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    device = _select_device(device_name)
    logger.info("Training MNIST example on %s (normalise=%s)", device, normalise)
    inputs, labels = torch.load(data_path, map_location="cpu", weights_only=True)
    inputs = inputs.float()
    if normalise:
        inputs = inputs / 255.0
    labels = labels.long()
    if learning_rate is None:
        learning_rate = 1e-3 if normalise else 1e-4

    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    count = inputs.shape[0]

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(count)
        total_loss = 0.0
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            batch_inputs = inputs[index].to(device)
            batch_labels = labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(index)
        logger.info("epoch %d/%d loss %.4f", epoch + 1, epochs, total_loss / count)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    config = {
        "benchmark": "mnist",
        "name": name or output_dir.name,
        "author": author or "unknown",
        "architecture": "mlp",
        "input_size": 784,
        "output_size": 10,
        "hidden_sizes": list(HIDDEN_SIZES),
        "activation": "relu",
        "normalisation": normalise,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("Wrote %s", output_dir / "model.pt")
    return state_dict


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/mnist-train.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mnist/example"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument(
        "--name", default=None, help="Display name (defaults to the output directory)"
    )
    parser.add_argument("--author", default="unknown", help="Author credited on the leaderboard")
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Scale pixels to [0, 1] instead of training on raw 0-255 inputs",
    )
    args = parser.parse_args(argv)
    train(
        args.data_path,
        args.output,
        epochs=args.epochs,
        device_name=args.device,
        normalise=args.normalise,
        name=args.name,
        author=args.author,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
