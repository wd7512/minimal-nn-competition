"""Answer format and the trusted checker.

A verifier answers each instance with JSON::

    {"id": "...", "answer": "PROVED",
     "leaves": [{"box": [[0, 1], [0]], "witness": 0, "method": "interval"},
                {"box": [[1], [1]], "witness": 0, "method": "point"}]}
    {"id": "...", "answer": "COUNTEREXAMPLE", "sequence": [0, 1]}
    {"id": "...", "answer": "UNKNOWN"}

A box lists the allowed tokens at each of the ``max_len`` input positions.
PROVED is accepted only when the leaves partition all inputs exactly and
every leaf holds:

- ``interval``: rigorous interval bounds show ``witness`` beats the forbidden
  token on every sequence in the box;
- ``point``: a single sequence where the float64 forward pass has
  ``logit_witness - logit_forbidden >= tau / 2``.

A COUNTEREXAMPLE is accepted when the float64 forward pass has the forbidden
token ahead of every other token by at least ``tau / 2``. Instances guarantee
every input is at least ``tau`` from a tie, far above float64 error.
No verifier code is run.
"""

import math

import numpy as np

from minimal_nn_competition.verification.forward import logits
from minimal_nn_competition.verification.interval import box_status

ANSWERS = ("PROVED", "COUNTEREXAMPLE", "UNKNOWN")
METHODS = ("interval", "point")


class CertificateError(ValueError):
    """Raised when an answer's claim is malformed or does not hold."""


def point_status(params, spec, forbidden, sequence, tau, witness=None):
    """Return ``("proved", j)``, ``("violated", None)`` or ``("open", None)`` for one sequence."""
    row = logits(params, spec, [tuple(sequence)])[0]
    others = [j for j in range(spec.vocab) if j != forbidden]
    if witness is None:
        witness = max(others, key=lambda j: row[j])
    if row[witness] - row[forbidden] >= tau / 2:
        return "proved", witness
    if row[forbidden] - max(row[j] for j in others) >= tau / 2:
        return "violated", None
    return "open", None


def _token(value, spec):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < spec.vocab:
        raise CertificateError(f"{value!r} is not a token in 0..{spec.vocab - 1}")
    return value


def _box(raw, spec):
    if not isinstance(raw, list) or len(raw) != spec.max_len:
        raise CertificateError(f"box must list exactly {spec.max_len} positions")
    box = []
    for position in raw:
        if not isinstance(position, list) or not position:
            raise CertificateError("each box position must be a non-empty list of tokens")
        tokens = [_token(token, spec) for token in position]
        if len(set(tokens)) != len(tokens):
            raise CertificateError("box positions must not repeat tokens")
        box.append(tokens)
    return box


def _check_partition(boxes, spec):
    """Leaves must cover all ``vocab ** max_len`` sequences exactly once."""
    covered = sum(math.prod(len(position) for position in box) for box in boxes)
    if covered != spec.vocab**spec.max_len:
        raise CertificateError(f"leaves cover {covered} of {spec.vocab**spec.max_len} sequences")
    masks = np.array(
        [[sum(1 << token for token in position) for position in box] for box in boxes],
        dtype=np.uint64,
    ).reshape(len(boxes), spec.max_len)
    # With the count equal, pairwise disjoint leaves cover every sequence.
    for i in range(len(boxes) - 1):
        if np.all((masks[i + 1 :] & masks[i]) != 0, axis=1).any():
            raise CertificateError("leaves overlap")


def _parse_leaf(leaf, spec, forbidden):
    if not isinstance(leaf, dict):
        raise CertificateError("each leaf must be an object")
    box, witness = _box(leaf.get("box"), spec), _token(leaf.get("witness"), spec)
    method = leaf.get("method", "interval")
    if witness == forbidden:
        raise CertificateError("a leaf witness must differ from the forbidden token")
    if method not in METHODS:
        raise CertificateError(f"leaf method must be one of {', '.join(METHODS)}")
    if method == "point" and any(len(position) != 1 for position in box):
        raise CertificateError("a point leaf must hold a single sequence")
    return box, witness, method


def _check_leaves(leaves, params, spec, forbidden, tau):
    """Check every leaf; point leaves share one batched forward pass."""
    points = []
    for leaf in leaves:
        box, witness, method = _parse_leaf(leaf, spec, forbidden)
        if method == "point":
            points.append((tuple(p[0] for p in box), witness))
        elif not box_status(params, spec, forbidden, box, witness)[0]:
            raise CertificateError(f"interval leaf {box} with witness {witness} does not hold")
    if points:
        rows = logits(params, spec, [sequence for sequence, _ in points])
        for (sequence, witness), row in zip(points, rows, strict=True):
            if row[witness] - row[forbidden] < tau / 2:
                raise CertificateError(f"point leaf {list(sequence)} with witness {witness} fails")
    _check_partition([_box(leaf["box"], spec) for leaf in leaves], spec)


def check_answer(params, spec, forbidden, tau, instance_id, answer):
    """Return the accepted answer kind, or raise CertificateError.

    ``params`` are float64 arrays. UNKNOWN makes no claim and is returned unchecked.
    """
    if not isinstance(answer, dict) or answer.get("id") != instance_id:
        raise CertificateError(f"answer must be a JSON object with id {instance_id!r}")
    kind = answer.get("answer")
    if kind not in ANSWERS:
        raise CertificateError(f"answer must be one of {', '.join(ANSWERS)}")
    if kind == "COUNTEREXAMPLE":
        sequence = answer.get("sequence")
        box = _box([[t] for t in sequence] if isinstance(sequence, list) else None, spec)
        if point_status(params, spec, forbidden, [p[0] for p in box], tau)[0] != "violated":
            raise CertificateError(f"{sequence} is not a counterexample")
    if kind == "PROVED":
        leaves = answer.get("leaves")
        if not isinstance(leaves, list) or not leaves:
            raise CertificateError("leaves must be a non-empty list")
        _check_leaves(leaves, params, spec, forbidden, tau)
    return kind
