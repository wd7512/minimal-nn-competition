"""Train the smallest 773-H-1 ReLU MLP fitting the fixed chess-eval-10k set.

The benchmark has no held-out split: submissions are scored on the same
committed 10,000 positions, and pass only when every prediction is within
0.1 pawns of Stockfish. This script trains single-hidden-layer MLPs of
increasing width with full-batch LBFGS, stops at the first width whose worst
absolute error is at most TARGET_MARGIN (a safety margin below the 0.1 pawn
tolerance), and writes config.json + model.pt for that width.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from minimal_nn_competition.benchmarks.chess_eval import (
    CHESS_EVAL_DATA,
    INPUT_SIZE,
    TOLERANCE_PAWNS,
    load_tensors,
)

TARGET_MARGIN = 0.08
WIDTHS = (20, 24, 28, 32, 36, 40)
PER_WIDTH_BUDGET_S = 420.0


def make_model(width):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_SIZE, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )


def max_abs_error(model, inputs, targets):
    with torch.no_grad():
        predictions = model(inputs).squeeze(1)
        return (predictions - targets).abs().max().item()


def train_width(width, inputs, targets):
    model = make_model(width)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=40,
        history_size=60,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )
    start = time.monotonic()
    steps = 0
    while time.monotonic() - start < PER_WIDTH_BUDGET_S:
        def closure():
            optimizer.zero_grad()
            loss = (model(inputs).squeeze(1) - targets).pow(2).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        steps += 1
        error = max_abs_error(model, inputs, targets)
        print(f"[width {width}] step {steps} max_err {error:.4f}", flush=True)
        if error <= TARGET_MARGIN:
            return model, error
    return None, max_abs_error(model, inputs, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs, targets = load_tensors(CHESS_EVAL_DATA["chess-eval-10k"])
    for width in WIDTHS:
        model, error = train_width(width, inputs, targets)
        if model is None:
            print(f"[width {width}] FAILED (max_err {error:.4f})", flush=True)
            continue
        name = f"glm-lbfgs-{width}"
        out_dir = args.output / name
        out_dir.mkdir(parents=True, exist_ok=True)
        state = model.state_dict()
        torch.save(state, out_dir / "model.pt")
        reloaded = torch.load(out_dir / "model.pt", weights_only=True)
        fresh = make_model(width)
        fresh.load_state_dict(reloaded)
        reload_error = max_abs_error(fresh, inputs, targets)
        if reload_error > TOLERANCE_PAWNS:
            print(f"[width {width}] reload check FAILED ({reload_error:.4f})", flush=True)
            continue
        params = int(sum(tensor.numel() for tensor in state.values()))
        config = {
            "benchmark": "chess-eval-10k",
            "name": name,
            "author": "GLM x le vibe",
            "architecture": "mlp",
            "input_size": INPUT_SIZE,
            "output_size": 1,
            "hidden_sizes": [width],
            "activation": "relu",
            "normalisation": False,
            "description": (
                f"Single-hidden-layer ReLU MLP ({INPUT_SIZE}-{width}-1) fitted to the "
                f"fixed public 10k set with full-batch LBFGS; {params} parameters, "
                f"max abs error {reload_error:.4f} pawns over all 10,000 positions."
            ),
            "input_indices": None,
        }
        (out_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        print(f"PASS name={name} params={params} max_err={reload_error:.4f}", flush=True)
        return
    print("NO WIDTH PASSED", flush=True)
    sys.exit(1)


main()
