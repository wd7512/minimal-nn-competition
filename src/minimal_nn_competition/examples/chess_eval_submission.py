"""Train a chess evaluation example submission on the committed Stockfish dataset."""

import argparse
import json
import logging
from pathlib import Path

INPUT_SIZE = 769
OUTPUT_SIZE = 1
HIDDEN_SIZES = (64, 64)


def create_model(hidden_sizes=HIDDEN_SIZES):
    """Return the MLP shape used by the example submission."""
    from torch import nn

    layers = []
    input_size = INPUT_SIZE
    for hidden_size in hidden_sizes:
        layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
        input_size = hidden_size
    layers.append(nn.Linear(input_size, OUTPUT_SIZE))
    return nn.Sequential(*layers)


def select_device(requested):
    """Resolve the requested torch device."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(
    output_dir,
    data_path=None,
    epochs=16_000,
    batch_size=256,
    learning_rate=1e-3,
    hidden_sizes=HIDDEN_SIZES,
    device_name="auto",
    seed=0,
    name=None,
    author="example author",
    benchmark="chess-eval-10k",
):
    """Train the example MLP and write ``config.json`` and ``model.pt``."""
    import torch
    from torch import nn

    from minimal_nn_competition.benchmarks.chess_eval import (
        CHESS_EVAL_DATA,
        TOLERANCE_PAWNS,
        load_tensors,
    )

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    device = select_device(device_name)
    logger.info(f"Training {benchmark} example on {device}")
    inputs, targets = load_tensors(data_path or CHESS_EVAL_DATA[benchmark])
    inputs = inputs.to(device)
    targets = targets.to(device)

    model = create_model(hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    count = len(inputs)
    best_state = None
    best_within = -1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        order = torch.randperm(count).to(device)
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs[index]).squeeze(1)
            loss = loss_fn(predictions, targets[index])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                predictions = model(inputs).squeeze(1)
                errors = (predictions - targets).abs()
                within = (errors <= TOLERANCE_PAWNS).float().mean().item()
                mae = errors.mean().item()
            logger.info(
                f"epoch {epoch + 1}/{epochs} loss {total_loss / count:.6f} "
                f"mae {mae:.4f} within {within:.6f}"
            )
            if within > best_within:
                best_within = within
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            if within >= 1.0:
                break
        scheduler.step()

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "model.pt")
    config = {
        "benchmark": benchmark,
        "name": name or output_dir.name,
        "author": author or "example author",
        "architecture": "mlp",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "hidden_sizes": list(hidden_sizes),
        "activation": "relu",
        "description": "Trained on the fixed Lichess Stockfish evaluation dataset.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"Wrote {output_dir / 'model.pt'} (best within {best_within:.6f})")
    return best_state


def main(argv=None):
    from minimal_nn_competition.benchmarks.chess_eval import CHESS_EVAL_DATA

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/chess-eval-10k/example"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(HIDDEN_SIZES))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--author", default="example author")
    parser.add_argument("--benchmark", default="chess-eval-10k", choices=tuple(CHESS_EVAL_DATA))
    args = parser.parse_args(argv)
    train(
        args.output,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=tuple(args.hidden_sizes),
        device_name=args.device,
        seed=args.seed,
        name=args.name,
        author=args.author,
        benchmark=args.benchmark,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
