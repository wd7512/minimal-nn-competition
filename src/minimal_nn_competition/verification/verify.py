"""Reference verifier: branch-and-bound over token sets with point checks at the leaves."""

import time

from minimal_nn_competition.verification.certificate import point_status
from minimal_nn_competition.verification.interval import box_status


def branch_and_bound(params, spec, forbidden, tau, timeout=60.0):
    """Answer one instance in the certificate format (without the ``id``).

    Start from the box holding every token at every position. A box the
    interval bounds prove becomes an ``interval`` leaf; otherwise its widest
    position is split in half. A single sequence is settled by the float64
    point check: a ``point`` leaf, a counterexample, or UNKNOWN on a near tie.
    """
    start = time.monotonic()
    leaves = []
    stack = [tuple(tuple(range(spec.vocab)) for _ in range(spec.max_len))]
    while stack:
        if time.monotonic() - start > timeout:
            return {"answer": "UNKNOWN"}
        box = stack.pop()
        proved, witness = box_status(params, spec, forbidden, box)
        if proved:
            leaves.append({"box": [list(p) for p in box], "witness": witness, "method": "interval"})
            continue
        widest = max(range(spec.max_len), key=lambda i: len(box[i]), default=None)
        if widest is None or len(box[widest]) == 1:
            sequence = [p[0] for p in box]
            status, witness = point_status(params, spec, forbidden, sequence, tau)
            if status == "violated":
                return {"answer": "COUNTEREXAMPLE", "sequence": sequence}
            if status == "open":
                return {"answer": "UNKNOWN"}
            leaves.append({"box": [[t] for t in sequence], "witness": witness, "method": "point"})
            continue
        half = len(box[widest]) // 2
        for part in (box[widest][half:], box[widest][:half]):
            stack.append(box[:widest] + (part,) + box[widest + 1 :])
    return {"answer": "PROVED", "leaves": leaves}
