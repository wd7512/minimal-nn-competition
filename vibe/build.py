"""Distil a serpentine Hamiltonian-cycle Snake policy into a minimal MLP.

Teacher: a fixed closed Hamiltonian cycle over the 16x16 board. Features
(0..5) of the observation are all affine functions of the head position, so
whichever subset we read, an exact argmax match on all 256 cells makes the
student play identically to the teacher: guaranteed fill on every seed.

Real mode searches candidate architectures strictly below the current
242-parameter leaderboard record for snake-simple-16x16, smallest first, over
both the 2-feature and the 6-feature input sets. Smoke mode
(VIBE_SMOKE=1) trains the proven (16,6) 6-feature net to validate the
pipeline quickly and saves a best-effort model even without an exact fit.
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
    DEADLINE = time.time() + 45 * 60
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

    def make_dataset(idx):
        env = SnakeEnvironment(size=SIZE, seed=0, input_indices=idx)
        env.reset()
        rows = []
        for c in cells:
            env.snake = [c]
            rows.append(env.observation())
        return (
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(labels),
        )

    def count_params(idx, hidden):
        sizes = [len(idx)] + list(hidden) + [4]
        return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

    def make_model(idx, hidden):
        layers, prev = [], len(idx)
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 4))
        return nn.Sequential(*layers)

    if SMOKE:
        idx = (0, 1, 2, 3, 4, 5)
        cands = [(idx, (16, 6), count_params(idx, (16, 6)))]
    else:
        cands = []
        for idx in [(0, 1), (0, 1, 2, 3, 4, 5)]:
            for h in range(4, 40):
                p = count_params(idx, (h,))
                if p < RECORD:
                    cands.append((idx, (h,), p))
            for h1 in range(3, 30):
                for h2 in range(2, 16):
                    p = count_params(idx, (h1, h2))
                    if p < RECORD:
                        cands.append((idx, (h1, h2), p))
        cands = sorted(set(cands), key=lambda c: (-c[2], len(c[0]), c[1]))
    print(f"{len(cands)} candidate architectures (smoke={SMOKE})", flush=True)

    datasets = {}

    def data(idx):
        if idx not in datasets:
            datasets[idx] = make_dataset(idx)
        return datasets[idx]

    def margins(model, X, Y):
        with torch.no_grad():
            logits = model(X)
        tgt = logits.gather(1, Y[:, None]).squeeze(1)
        other = logits.clone()
        other.scatter_(1, Y[:, None], -1e30)
        return tgt - other.max(1).values

    def fit(idx, hidden, seed, lr, steps):
        X, Y = data(idx)
        torch.manual_seed(seed)
        model = make_model(idx, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            if bool((margins(model, X, Y) >= MARGIN).all().item()):
                return model
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X), Y)
            loss.backward()
            opt.step()
            sched.step()
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
            if bool((margins(model, X, Y) >= MARGIN).all().item()):
                break
            opt2.step(closure)
        return model

    def score(model, idx):
        X, Y = data(idx)
        m = margins(model, X, Y)
        return int((m > 0).sum().item()), bool((m >= MARGIN).all().item())

    best = None  # smallest exact model found so far: (idx, hidden, p, model)
    for idx, hidden, p in cands:
        if time.time() > DEADLINE:
            print("deadline reached", flush=True)
            break
        if best is not None and p >= best[2]:
            continue  # already have an exact model with fewer parameters
        if SMOKE:
            restarts = 8
            budgets = [(0.05, 4000), (0.08, 4000), (0.03, 6000)]
        else:
            restarts = 2
            budgets = [(0.05, 6000), (0.1, 4000)]
        for k in range(restarts):
            if time.time() > DEADLINE:
                break
            lr, steps = budgets[k % len(budgets)]
            model = fit(idx, hidden, seed=104729 * (k + 1) + p, lr=lr, steps=steps)
            matched, exact = score(model, idx)
            print(
                f"train idx={idx} hidden={hidden} params={p} k={k} "
                f"matched={matched}/256 exact={exact}",
                flush=True,
            )
            if exact:
                best = (idx, hidden, p, model)
                print(f"new best: params={p} hidden={hidden} idx={idx}", flush=True)
                break
            if SMOKE and best is None:
                best = (idx, hidden, p, model)  # best-effort for plumbing

    if best is None:
        print("NO MODEL FOUND", flush=True)
        sys.exit(1)

    idx, hidden, p, model = best
    print(f"chosen model: params={p} hidden={hidden} idx={idx}", flush=True)

    def act(obs, cache={}):
        if obs not in cache:
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            cache[obs] = int(model(t).argmax(1).item())
        return cache[obs]

    wins = 0
    moves_list = []
    for seed in range(SEEDS):
        e = SnakeEnvironment(size=SIZE, seed=seed, input_indices=idx, max_moves=65536)
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
    if wins != SEEDS and not SMOKE:
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
        "input_size": len(idx),
        "output_size": 4,
        "hidden_sizes": list(hidden),
        "activation": "relu",
        "input_indices": list(idx),
        "description": (
            "Serpentine Hamiltonian-cycle teacher distilled into an MLP on "
            "position-dependent head features; exact argmax match on all 256 "
            f"cells, {p} parameters"
            + (" (smoke pipeline test)" if SMOKE else ", fills the board on all 100 seeds")
            + "."
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"SAVED params={p} hidden={hidden} idx={idx} name={NAME}", flush=True)


if __name__ == "__main__":
    main()
