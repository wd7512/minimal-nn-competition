"""Shared supervised training loop for 28x28 image-classification examples."""

import argparse
import json
import logging
from pathlib import Path

HIDDEN = 64
HIDDEN_SIZES = (HIDDEN,)


def build_model(input_size=784, output_size=10):
    """Return the MLP shape used by the image-classification examples."""
    from torch import nn

    return nn.Sequential(nn.Linear(input_size, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, output_size))


def select_device(requested):
    """Resolve the requested torch device."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_classifier(
    data_path,
    output_dir,
    benchmark,
    epochs=20,
    batch_size=128,
    learning_rate=None,
    device_name="auto",
    seed=0,
    normalise=False,
    name=None,
    author="unknown",
):
    """Train the example MLP and write ``config.json`` and ``model.pt``."""
    import torch

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    device = select_device(device_name)
    logger.info(f"Training {benchmark} example on {device} (normalise={normalise})")
    inputs, labels = torch.load(data_path, map_location="cpu", weights_only=True)
    inputs = inputs.float()
    if normalise:
        inputs = inputs / 255.0
    labels = labels.long()
    if learning_rate is None:
        learning_rate = 1e-3 if normalise else 1e-4

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = len(inputs)

    for epoch in range(epochs):
        total_loss = 0.0
        for index in torch.randperm(count).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(inputs[index].to(device)), labels[index].to(device)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)
        logger.info(f"epoch {epoch + 1}/{epochs} loss {total_loss / count:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.cpu()
    state_dict = model.state_dict()
    model_path = output_dir / "model.pt"
    torch.save(state_dict, model_path)
    config = dict(
        benchmark=benchmark,
        name=name or output_dir.name,
        author=author or "unknown",
        architecture="mlp",
        input_size=784,
        output_size=10,
        hidden_sizes=[HIDDEN],
        activation="relu",
        normalisation=normalise,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"Wrote {model_path}")
    return state_dict


def run(display_name, data_path, output, argv=None):
    """Run the shared training CLI for one image-classification example."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=f"Train a small {display_name} classifier and write a submission."
    )
    add = parser.add_argument
    add("--data-path", type=Path, default=Path(data_path))
    add("--output", dest="output_dir", type=Path, default=Path(output))
    add("--epochs", type=int, default=20)
    add("--device", dest="device_name", default="auto", choices=("auto", "mps", "cpu"))
    add("--name", default=None, help="Display name (defaults to the output directory)")
    add("--author", default="unknown", help="Author credited on the leaderboard")
    add(
        "--normalise",
        action="store_true",
        help="Scale pixels to [0, 1] instead of training on raw 0-255 inputs",
    )
    args = parser.parse_args(argv)
    train_classifier(**vars(args), benchmark=display_name.lower())
    return 0
