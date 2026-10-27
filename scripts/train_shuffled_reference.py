"""Train a small example for either shuffled-label image benchmark."""

import argparse
import logging
from pathlib import Path

from minimal_nn_competition.examples.classifier_submission import train_classifier


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("mnist-shuffled", "fashion-mnist-shuffled"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_classifier(
        data_path=args.data_path,
        output_dir=args.output,
        benchmark=args.benchmark,
        epochs=args.epochs,
        batch_size=128,
        learning_rate=1e-3,
        seed=0,
        normalise=True,
        name="pilot-64x20",
        author="issue-191 pilot",
    )


if __name__ == "__main__":
    main()
