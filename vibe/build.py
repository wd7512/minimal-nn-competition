"""Beat the snake-simple-16x16 record by pruning the leader's exact model.

The leader's 242-parameter model (cycle-16x6, input features 0-5) is a
position-only policy: features 0-5 are affine in the head position, so its
argmax is a function of the head cell alone. We distil its argmax on all 256
cells and then greedily prune input features and hidden units, finetuning
with LBFGS after every removal. Any exact match with fewer parameters beats
the record; identical argmax on every cell means identical gameplay.
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
SMOKE = os.environ.get("VIBE_SMOKE", "1") == "1"
MARGIN = 1e-3

if SMOKE:
    NAME = "glm-cycle-smoke"
    DEADLINE = time.time() + 10 * 60
    SEEDS = 5
else:
    NAME = "glm-cycle-distilled"
    DEADLINE = time.time() + 45 * 60
    SEEDS = 100

TEACHER = Path("models/snake-simple-16x16/cycle-16x6")


def log(msg):
    print(msg, flush=True)


def obs_matrix(idx):
    env = SnakeEnvironment(size=SIZE, seed=0, input_indices=idx)
    env.reset()
    rows = []
    for y in range(SIZE):
        for x in range(SIZE):
            env.snake = [(x, y)]
            rows.append(env.observation())
    return torch.tensor(rows, dtype=torch.float32)


def mlp_forward(X, layers):
    # layers: list of (weight, bias) pairs with relu between, none at the end
    h = X
    for i, (w, b) in enumerate(layers):
        if i:
            h = torch.relu(h)
        h = h @ w.T + b
    return h


def main():
    t0 = time.time()

    state = torch.load(TEACHER / "model.pt", weights_only=True)
    cfg = json.loads((TEACHER / "config.json").read_text())
    full_idx = list(cfg["input_indices"])
    X_full = obs_matrix(full_idx)
    W = [
        (state["0.weight"], state["0.bias"]),
        (state["2.weight"], state["2.bias"]),
        (state["4.weight"], state["4.bias"]),
    ]
    with torch.no_grad():
        T_logits = mlp_forward(X_full, W)
    T_labels = T_logits.argmax(1)
    tgt = T_logits.gather(1, T_labels[:, None]).squeeze(1)
    other = T_logits.clone()
    other.scatter_(1, T_labels[:, None], -1e30)
    T_margins = tgt - other.max(1).values
    log(f"teacher labels: {dict(zip(*T_labels.unique(return_counts=True)))}")
    log(f"teacher min margin {T_margins.min().item():.6g}")

    # current pruned model: input indices + three layers
    model = {
        "idx": list(full_idx),
        "W1": W[0][0].clone(), "b1": W[0][1].clone(),
        "W2": W[1][0].clone(), "b2": W[1][1].clone(),
        "W3": W[2][0].clone(), "b3": W[2][1].clone(),
    }

    def param_count(m):
        n = len(m["idx"]) * m["W1"].shape[0] + m["b1"].shape[0]
        n += m["W1"].shape[0] * m["W2"].shape[0] + m["b2"].shape[0]
        n += m["W2"].shape[0] * m["W3"].shape[0] + m["b3"].shape[0]
        return n

    def evaluate(m, train=None):
        """Return (matched, exact, min_margin) against teacher labels."""
        if train is None:
            train = m
        X = obs_matrix(m["idx"])
        with torch.no_grad():
            logits = mlp_forward(X, [(train["W1"], train["b1"]),
                                     (train["W2"], train["b2"]),
                                     (train["W3"], train["b3"])])
        pred = logits.argmax(1)
        matched = int((pred == T_labels).sum().item())
        tgt = logits.gather(1, T_labels[:, None]).squeeze(1)
        o = logits.clone()
        o.scatter_(1, T_labels[:, None], -1e30)
        mm = (tgt - o.max(1).values).min().item()
        return matched, matched == len(T_labels), mm

    def finetune(m, rounds=6):
        """LBFGS finetune of a (possibly reduced) model; returns best model."""
        params = [m["W1"], m["b1"], m["W2"], m["b2"], m["W3"], m["b3"]]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=100,
                                line_search_fn="strong_wolfe")
        X = obs_matrix(m["idx"])
        Yt = T_labels

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(mlp_forward(X, [
                (m["W1"], m["b1"]), (m["W2"], m["b2"]), (m["W3"], m["b3"]),
            ]), Yt)
            loss.backward()
            return loss

        best = None
        for _ in range(rounds):
            _, exact, mm = evaluate(m)
            if exact and mm >= MARGIN:
                return _copy(m)
            if best is None or mm > best[0]:
                best = (mm, _copy(m))
            opt.step(closure)
        _, exact, mm = evaluate(m)
        if exact and mm >= MARGIN:
            return _copy(m)
        return best[1] if best else None

    matched, exact, mm = evaluate(model)
    log(f"start: params={param_count(model)} matched={matched}/256 "
        f"exact={exact} min_margin={mm:.4g}")
    if not exact:
        log("teacher-clone failed; aborting", flush=True)
        sys.exit(1)

    def _copy(m):
        return {k: (v.detach().clone() if torch.is_tensor(v) else list(v))
                for k, v in m.items()}

    def drop_input(m, j):
        n = _copy(m)
        n["idx"] = m["idx"][:j] + m["idx"][j + 1:]
        n["W1"] = torch.cat([m["W1"][:, :j], m["W1"][:, j + 1:]], dim=1).clone()
        return n

    def drop_l1(m, j):
        n = _copy(m)
        n["W1"] = torch.cat([m["W1"][:j], m["W1"][j + 1:]]).clone()
        n["b1"] = torch.cat([m["b1"][:j], m["b1"][j + 1:]]).clone()
        n["W2"] = torch.cat([m["W2"][:, :j], m["W2"][:, j + 1:]], dim=1).clone()
        return n

    def drop_l2(m, j):
        n = _copy(m)
        n["W2"] = torch.cat([m["W2"][:j], m["W2"][j + 1:]]).clone()
        n["b2"] = torch.cat([m["b2"][:j], m["b2"][j + 1:]]).clone()
        n["W3"] = torch.cat([m["W3"][:, :j], m["W3"][:, j + 1:]], dim=1).clone()
        return n

    def drop_output(m, j):
        n = _copy(m)
        n["W3"] = torch.cat([m["W3"][:j], m["W3"][j + 1:]]).clone()
        n["b3"] = torch.cat([m["b3"][:j], m["b3"][j + 1:]]).clone()
        return n

    drop = {
        "input": (lambda m: len(m["idx"]), lambda m, j: drop_input(m, j)),
        "l1": (lambda m: m["W1"].shape[0], lambda m, j: drop_l1(m, j)),
        "l2": (lambda m: m["W2"].shape[0], lambda m, j: drop_l2(m, j)),
    }

    # Output classes cannot be dropped: the engine needs all 4 actions? It
    # requests argmax over a 4-logit vector, so output size must stay 4.
    # But if the teacher never uses some action, dropping is still unsafe
    # (validator expects output_size 4), so skip output pruning.

    step = 0

    def try_op(m, kind, j, rounds=6, tag=""):
        """Try removing one unit; returns reduced model or None."""
        size, rem = drop[kind]
        if kind == "input" and len(m["idx"]) <= 1:
            return None
        if size(m) <= j:
            return None
        cand = rem(m, j)
        out = finetune(cand, rounds=rounds)
        if out is not None:
            _, exact, mm = evaluate(out)
            if exact and mm >= MARGIN:
                return out
        return None

    def current_ops(m):
        ops = []
        for j in range(len(m["idx"])):
            ops.append(("input", j))
        for j in range(m["W1"].shape[0]):
            ops.append(("l1", j))
        for j in range(m["W2"].shape[0]):
            ops.append(("l2", j))
        return ops

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # phase 1: greedy single removals
        for kind, j in current_ops(model):
            if time.time() > DEADLINE:
                break
            out = try_op(model, kind, j)
            step += 1
            if out is not None:
                model = out
                improved = True
                log(f"[{time.time()-t0:6.1f}s] pruned {kind}[{j}] -> "
                    f"params={param_count(model)}")
            else:
                log(f"[{time.time()-t0:6.1f}s] keep {kind}[{j}] "
                    f"(trial {step} failed)")
        if improved or time.time() > DEADLINE:
            continue
        # phase 2: simultaneous pair removals
        ops = current_ops(model)
        n = len(ops)
        pair_done = 0
        for a in range(n):
            if time.time() > DEADLINE or improved:
                break
            for b in range(a + 1, n):
                if time.time() > DEADLINE:
                    break
                ka, ia = ops[a]
                kb, ib = ops[b]
                _, rem_a = drop[ka]
                _, rem_b = drop[kb]
                if ka == kb and ia >= ib:
                    continue
                if ka == kb:
                    # same layer: remove the higher index first
                    c1 = rem_b(model, ib)
                    cand = rem_a(c1, ia)
                else:
                    c1 = rem_a(model, ia)
                    cand = rem_b(c1, ib)
                step += 1
                pair_done += 1
                out = finetune(cand, rounds=14)
                if out is not None:
                    _, exact, mm = evaluate(out)
                    if exact and mm >= MARGIN:
                        model = out
                        improved = True
                        log(f"[{time.time()-t0:6.1f}s] pruned pair "
                            f"{ka}[{ia}]+{kb}[{ib}] -> "
                            f"params={param_count(model)}")
                        break
        if pair_done and not improved:
            log(f"[{time.time()-t0:6.1f}s] all {pair_done} pairs failed")

    matched, exact, mm = evaluate(model)
    p = param_count(model)
    log(f"final: params={p} hidden=({model['W1'].shape[0]},{model['W2'].shape[0]}) "
        f"idx={model['idx']} matched={matched}/256 min_margin={mm:.4g}")
    if p >= 242 and not SMOKE:
        log("did not beat the 242 record; aborting", flush=True)
        sys.exit(1)

    # gameplay check on real trajectories
    Wm = [(model["W1"], model["b1"]), (model["W2"], model["b2"]),
          (model["W3"], model["b3"])]

    def act(obs, cache={}):
        if obs not in cache:
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                cache[obs] = int(mlp_forward(t, Wm).argmax(1).item())
        return cache[obs]

    wins = 0
    moves_list = []
    for seed in range(SEEDS):
        e = SnakeEnvironment(size=SIZE, seed=seed,
                             input_indices=model["idx"], max_moves=65536)
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
    log(f"wins={wins}/{SEEDS} mean_moves={statistics.fmean(moves_list):.1f}")
    if wins != SEEDS and not SMOKE:
        sys.exit(2)

    out = Path("models") / BENCH / NAME
    out.mkdir(parents=True, exist_ok=True)
    state_out = {
        "0.weight": model["W1"].detach().to(torch.float32),
        "0.bias": model["b1"].detach().to(torch.float32),
        "2.weight": model["W2"].detach().to(torch.float32),
        "2.bias": model["b2"].detach().to(torch.float32),
        "4.weight": model["W3"].detach().to(torch.float32),
        "4.bias": model["b3"].detach().to(torch.float32),
    }
    torch.save(state_out, out / "model.pt")
    config = {
        "benchmark": BENCH,
        "name": NAME,
        "author": AUTHOR,
        "architecture": "mlp",
        "input_size": len(model["idx"]),
        "output_size": 4,
        "hidden_sizes": [model["W1"].shape[0], model["W2"].shape[0]],
        "activation": "relu",
        "input_indices": model["idx"],
        "description": (
            f"Pruned and finetuned from the cycle-16x6 teacher: {p} "
            "parameters, exact argmax match on all 256 head cells, fills the "
            "board on all 100 seeds."
            if not SMOKE else
            f"Pipeline smoke test, {p} parameters."
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"SAVED params={p} hidden={config['hidden_sizes']} name={NAME}")


if __name__ == "__main__":
    main()
