"""Diagnostic: inspect the current leader's exact 242-param model."""

import json
from pathlib import Path

import torch

SIZE = 16
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ARROW = {0: "U", 1: "R", 2: "D", 3: "L"}


def serpentine_next(size):
    cyc = [(x, 0) for x in range(size)]
    for y in range(1, size):
        xs = range(size - 1, 0, -1) if y % 2 else range(1, size)
        cyc += [(x, y) for x in xs]
    cyc.append((0, size - 1))
    cyc += [(0, y) for y in range(size - 2, 0, -1)]
    return {a: cyc[(i + 1) % len(cyc)] for i, a in enumerate(cyc)}


def main():
    d = Path("models/snake-simple-16x16/cycle-16x6")
    print("=== config ===", flush=True)
    print((d / "config.json").read_text(), flush=True)
    state = torch.load(d / "model.pt", weights_only=True)
    print("=== state dict ===", flush=True)
    for k, v in state.items():
        t = v.detach().to(torch.float64)
        print(k, list(t.shape))
        if t.ndim == 1:
            print("[" + ", ".join(f"{x:.6g}" for x in t.tolist()) + "]")
        else:
            for row in t.tolist():
                print("[" + ", ".join(f"{x:.6g}" for x in row) + "]")
    print("=== argmax grid (rows=y=0 top) ===", flush=True)
    nxt = serpentine_next(SIZE)
    cells = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    labels = [DIRECTIONS.index((nxt[c][0] - c[0], nxt[c][1] - c[1])) for c in cells]
    from minimal_nn_competition.benchmarks.common.snake.environment import (
        SnakeEnvironment,
    )
    idx = tuple(json.loads((d / "config.json").read_text())["input_indices"])
    env = SnakeEnvironment(size=SIZE, seed=0, input_indices=idx)
    env.reset()
    rows = []
    for c in cells:
        env.snake = [c]
        rows.append(env.observation())
    X = torch.tensor(rows, dtype=torch.float32)
    logits = torch.nn.functional.linear(
        torch.nn.functional.relu(
            torch.nn.functional.linear(X, state["0.weight"], state["0.bias"])
        ),
        state["2.weight"],
        state["2.bias"],
    )
    logits = torch.nn.functional.linear(
        torch.nn.functional.relu(
            torch.nn.functional.linear(logits, state["4.weight"], state["4.bias"])
        ),
        state["6.weight"],
        state["6.bias"],
    )
    arg = logits.argmax(1).tolist()
    tgt = torch.tensor(logits).gather(1, torch.tensor(labels)[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, torch.tensor(labels)[:, None], -1e30)
    margins = (tgt - other.max(1).values).tolist()
    grid = [arg[y * SIZE:(y + 1) * SIZE] for y in range(SIZE)]
    for y, row in enumerate(grid):
        print("".join(ARROW[a] for a in row) + "   margin_min="
              + f"{min(margins[y*SIZE:(y+1)*SIZE]):.4g}", flush=True)
    matched = sum(a == l for a, l in zip(arg, labels))
    print(f"matched serpentine teacher on {matched}/256 cells", flush=True)
    print(f"global min margin {min(margins):.6g}", flush=True)


if __name__ == "__main__":
    main()
