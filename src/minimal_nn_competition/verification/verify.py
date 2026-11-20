"""Reference verifier (branch-and-bound over token sets) and exhaustive oracle."""

import time
from dataclasses import dataclass, field

from minimal_nn_competition.verification.interval import bound_logits, box_status
from minimal_nn_competition.verification.reference import all_sequences, input_count

MAX_ORACLE_INPUTS = 100_000


@dataclass
class Result:
    status: str  # PROVED, COUNTEREXAMPLE or UNKNOWN
    leaves: list = field(default_factory=list)  # (box, witness) pairs when PROVED
    counterexample: tuple | None = None
    boxes: int = 0
    seconds: float = 0.0
    reason: str = ""


def branch_and_bound(params, spec, forbidden, timeout=60.0, max_leaves=10_000):
    """Prove ``forbidden`` is never the next token, or find a sequence where it is.

    For each length, start from the box holding every token at every position.
    A box the interval bounds prove becomes a certificate leaf; otherwise its
    widest position is split in half. A single sequence that is neither proved
    nor violated (a near tie) gives UNKNOWN.
    """
    start = time.monotonic()
    leaves, boxes = [], 0

    def finish(status, **kwargs):
        return Result(status, boxes=boxes, seconds=time.monotonic() - start, **kwargs)

    for length in range(spec.max_len + 1):
        stack = [tuple(tuple(range(spec.vocab)) for _ in range(length))]
        while stack:
            if time.monotonic() - start > timeout:
                return finish("UNKNOWN", reason="timeout")
            if len(leaves) >= max_leaves:
                return finish("UNKNOWN", reason="leaf limit")
            box = stack.pop()
            boxes += 1
            status, witness = box_status(params, spec, forbidden, box)
            if status == "proved":
                leaves.append((box, witness))
                continue
            if status == "violated":
                return finish("COUNTEREXAMPLE", counterexample=tuple(p[0] for p in box))
            widest = max(range(length), key=lambda i: len(box[i]), default=None)
            if widest is None or len(box[widest]) == 1:
                return finish("UNKNOWN", reason=f"near tie on {[p[0] for p in box]}")
            half = len(box[widest]) // 2
            for part in (box[widest][half:], box[widest][:half]):
                stack.append(box[:widest] + (part,) + box[widest + 1 :])
    return finish("PROVED", leaves=leaves)


def oracle(params, spec, forbidden, model_name):
    """Ground truth by checking every input separately (small input spaces only)."""
    if input_count(spec) > MAX_ORACLE_INPUTS:
        raise ValueError(f"oracle supports at most {MAX_ORACLE_INPUTS} inputs")
    rows, answer, witness = [], "PROVED", None
    for seq in all_sequences(spec):
        lo, hi = bound_logits(params, spec, [[token] for token in seq])
        mid = (lo + hi) / 2
        best = max((j for j in range(spec.vocab) if j != forbidden), key=lambda j: lo[j])
        rows.append(
            {
                "seq": list(seq),
                "argmax": int(mid.argmax()),
                "margin": float(lo[best] - hi[forbidden]),
            }
        )
        status, _ = box_status(params, spec, forbidden, [[token] for token in seq])
        # A counterexample anywhere outranks an earlier near tie.
        if (status == "violated" and answer != "COUNTEREXAMPLE") or (
            status == "open" and answer == "PROVED"
        ):
            answer = "COUNTEREXAMPLE" if status == "violated" else "UNKNOWN"
            witness = list(seq)
    result = {"model_id": model_name, "forbidden": forbidden, "answer": answer, "inputs": rows}
    return result | ({"witness": witness} if witness is not None else {})
