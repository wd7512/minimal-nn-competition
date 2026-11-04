"""Distil a serpentine Hamiltonian-cycle Snake policy into a minimal MLP.

Teacher: a fixed closed Hamiltonian cycle over the 16x16 board. The student
MLP reads only observation indices (0, 1) (head distances), which depend
purely on the head position, so an exact argmax match on all 256 cells makes
the student play identically to the teacher: guaranteed fill on every seed.

Real mode searches architectures strictly below the current 242-parameter
leaderboard record for snake-simple-16x16. Smoke mode (VIBE_SMOKE=1) trains
one generously sized net to validate the entire pipeline quickly.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

from minimal_nn_competition.benchmarks.common.snake.environment import (
    SnakeEnvironment,
)

SIZE = 16
BENCH = "snake-simple-16x16"
AUTHOR = "GLM x le vibe"
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
IDX = (0, 1)
MARGIN = 1e-3
SMOKE = os.environ.get("VIBE_SMOKE", "1") == "1"

if SMOKE:
    NAME = "glm-cycle-smoke"
    RECORD = 10 ** 9
    DEADLINE = time.time() + 10 * 60
    SEEDS = 5
else:
    NAME = "glm-cycle-distilled"
    RECORD = 242  # current leaderboard record; require strictly fewer parameters
    DEADLINE = time.time() + 40 * 60
    SEEDS = 100


def serpentine_next(size):
    cyc = [(x, 0) for x in range(size)]
    for y in range(1, size):
        xs = range(size - 1, 0, -1) if y % 2 else range(1, size)
        cyc += [(x, y) for x in xs]
    cyc.append((0, size - 1))
    cyc += [(0, y) for y in range(size - 2, 0, -1)]
    assert len(cyc) == size * size == len(set(cyc)), len(cyc)
    for i in range(len(cyc)):
        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    return {a: cyc[(i + 1) % len(cyc)] for i, a in enumerate(cyc)}


def main():
    nxt = serpentine_next(SIZE)
    cells = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    labels = [
        DIRECTIONS.index((nxt[c][0] - c[0], nxt[c][1] - c[1])) for c in cells
    ]
    start = (SIZE // 2, SIZE // 2)
    first = DIRECTIONS[labels[cells.index(start)]]
    assert first != (-1, 0), "teacher would request a reversal on move one"

    env = SnakeEnvironment(size=SIZE, seed=0, input_indices=IDX)
    env.reset()
    rows = []
    for c in cells:
        env.snake = [c]
        rows.append(env.observation())
    X = torch.tensor(rows, dtype=torch.float32)
    Y = torch.tensor(labels)
    n = len(cells)

    def count_params(hidden):
        sizes = [len(IDX)] + list(hidden) + [4]
        return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

    def make_model(hidden):
        layers, prev = [], len(IDX)
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 4))
        return nn.Sequential(*layers)

    def margins(model):
        with torch.no_grad():
            logits = model(X)
        tgt = logits.gather(1, Y[:, None]).squeeze(1)
        other = logits.clone()
        other.scatter_(1, Y[:, None], -1e30)
        return tgt - other.max(1).values

    def fit(hidden, seed, lr, steps):
        torch.manual_seed(seed)
        model = make_model(hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            m = margins(model)
            if bool((m >= MARGIN).all().item()):
                return model
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X), Y)
            loss.backward()
            opt.step()
            sched.step()
        # LBFGS polish: often drives the last few mismatches to exact fit.
        opt2 = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=150,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt2.zero_grad()
            loss = nn.functional.cross_entropy(model(X), Y)
            loss.backward()
            return loss

        for _ in range(4):
            if bool((margins(model) >= MARGIN).all().item()):
                break
            opt2.step(closure)
        return model

    if SMOKE:
        cands = [(count_params((64, 32)), (64, 32))]
    else:
        cands = []
        for h in range(8, 34):
            p = count_params((h,))
            if p < RECORD:
                cands.append((p, (h,)))
        for h1 in range(4, 25):
            for h2 in range(2, 13):
                p = count_params((h1, h2))
                if p < RECORD:
                    cands.append((p, (h1, h2)))
        cands = sorted(set(cands))
    print(f"{len(cands)} candidate architectures (smoke={SMOKE})", flush=True)

    best = None
    if SMOKE:
        for p, hidden in cands:
            model = fit(hidden, seed=1, lr=0.05, steps=6000)
            m = margins(model)
            matched = int((m > 0).sum().item())
            print(f"smoke params={p} hidden={hidden} matched={matched}/{n}", flush=True)
            if matched == n:
                best = (p, hidden, model)
                break
    else:
        scores = []
        for p, hidden in cands:
            if time.time() > DEADLINE:
                break
            model = fit(hidden, seed=p, lr=0.05, steps=1800)
            matched = int((margins(model) > 0).sum().item())
            scores.append((p, hidden, matched))
            print(f"screen params={p} hidden={hidden} matched={matched}/{n}", flush=True)
        for threshold in (224, 192, 160):
            if best or time.time() > DEADLINE:
                break
            promising = sorted((p, h) for p, h, m in scores if m >= threshold)
            print(f"deep pass threshold={threshold}: {len(promising)} candidates", flush=True)
            for p, hidden in promising:
                if best or time.time() > DEADLINE:
                    break
                for k, (lr, steps) in enumerate(
                    [(0.05, 9000), (0.01, 12000), (0.03, 9000)]
                ):
                    if time.time() > DEADLINE:
                        break
                    model = fit(hidden, seed=7919 * (k + 1) + p, lr=lr, steps=steps)
                    m = margins(model)
                    matched = int((m > 0).sum().item())
                    print(f"deep params={p} hidden={hidden} lr={lr} matched={matched}/{n}", flush=True)
                    if (m >= MARGIN).all().item():
                        best = (p, hidden, model)
                        break
            if best:
                break

    if best is None:
        print("NO EXACT MODEL FOUND", flush=True)
        sys.exit(1)

    p, hidden, model = best
    print(f"exact model: params={p} hidden={hidden}", flush=True)

    def act(obs, cache={}):
        if obs not in cache:
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            cache[obs] = int(model(t).argmax(1).item())
        return cache[obs]

    wins = 0
    moves_list = []
    for seed in range(SEEDS):
        e = SnakeEnvironment(size=SIZE, seed=seed, input_indices=IDX, max_moves=65536)
        obs = e.reset()
        done = False
        last = len(e.snake)
        since_eat = 0
        while not done:
            obs, _, done = e.step(act(obs))
            if len(e.snake) > last:
                last = len(e.snake)
                since_eat = 0
            else:
                since_eat += 1
                if since_eat > 512:
                    done = True
        wins += bool(e.won)
        moves_list.append(e.moves)
    print(f"wins={wins}/{SEEDS} mean_moves={statistics.fmean(moves_list):.1f}", flush=True)
    if wins != SEEDS:
        sys.exit(2)

    out = Path("models") / BENCH / NAME
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    torch.save(state, out / "model.pt")
    config = {
        "benchmark": BENCH,
        "name": NAME,
        "author": AUTHOR,
        "architecture": "mlp",
        "input_size": len(IDX),
        "output_size": 4,
        "hidden_sizes": list(hidden),
        "activation": "relu",
        "input_indices": list(IDX),
        "description": (
            "Serpentine Hamiltonian-cycle teacher distilled into an MLP on the "
            "two head-distance features; exact argmax match on all 256 cells, "
            f"{p} parameters"
            + (" (smoke pipeline test)" if SMOKE else ", fills the board on all 100 seeds")
            + "."
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"SAVED params={p} hidden={hidden} name={NAME}", flush=True)


if __name__ == "__main__":
    main()
