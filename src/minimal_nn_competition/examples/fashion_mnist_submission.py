"""Train a small Fashion-MNIST classifier and write a submission."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from minimal_nn_competition.examples.classifier_submission import (
    build_model,
    train_classifier,
)


def create_model() -> Any:
    """Return the MLP shape used by the example submission."""
    return build_model()


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
    """Train the example Fashion-MNIST MLP and write ``config.json`` and ``model.pt``."""
    return train_classifier(
        data_path,
        output_dir,
        benchmark="fashion-mnist",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device_name=device_name,
        seed=seed,
        normalise=normalise,
        name=name,
        author=author,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/fashion-mnist-train.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/fashion-mnist/example"))
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
