"""Certificate format and the trusted checker.

A verifier answers each instance with JSON::

    {"format": 1, "model_id": "...", "forbidden": 1, "answer": "PROVED",
     "leaves": [{"box": [[0, 1], [0]], "witness": 0}, ...]}
    {"format": 1, "model_id": "...", "forbidden": 1, "answer": "COUNTEREXAMPLE",
     "sequence": [0, 1]}
    {"format": 1, "model_id": "...", "forbidden": 1, "answer": "UNKNOWN"}

A box lists the allowed tokens at each position, so its length is the
sequence length. PROVED is accepted only when, for every length, the leaves
partition the input space exactly and every leaf passes the interval check
with its witness. No verifier code is run.
"""

import math

import numpy as np

from minimal_nn_competition.verification.interval import box_status

FORMAT = 1
MAX_LEAVES = 10_000
ANSWERS = ("PROVED", "COUNTEREXAMPLE", "UNKNOWN")


class CertificateError(ValueError):
    """Raised when a certificate's claim is malformed or does not hold."""


def make_certificate(result, model_name, forbidden):
    cert = {"format": FORMAT, "model_id": model_name, "forbidden": forbidden}
    if result.status == "PROVED":
        leaves = [{"box": [list(p) for p in box], "witness": w} for box, w in result.leaves]
        return cert | {"answer": "PROVED", "leaves": leaves}
    if result.status == "COUNTEREXAMPLE":
        return cert | {"answer": "COUNTEREXAMPLE", "sequence": list(result.counterexample)}
    return cert | {"answer": "UNKNOWN"}


def _token(value, spec):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < spec.vocab:
        raise CertificateError(f"{value!r} is not a token in 0..{spec.vocab - 1}")
    return value


def _box(raw, spec):
    if not isinstance(raw, list) or len(raw) > spec.max_len:
        raise CertificateError(f"box must be a list of at most {spec.max_len} positions")
    box = []
    for position in raw:
        if not isinstance(position, list) or not position:
            raise CertificateError("each box position must be a non-empty list of tokens")
        tokens = [_token(token, spec) for token in position]
        if len(set(tokens)) != len(tokens):
            raise CertificateError("box positions must not repeat tokens")
        box.append(tokens)
    return box


def _check_partition(boxes, spec, length):
    """Leaves of one length must cover all vocab**length sequences exactly once."""
    covered = sum(math.prod(len(position) for position in box) for box in boxes)
    if covered != spec.vocab**length:
        raise CertificateError(
            f"length {length}: leaves cover {covered} of {spec.vocab**length} sequences"
        )
    masks = np.array(
        [[sum(1 << token for token in position) for position in box] for box in boxes],
        dtype=np.uint64,
    ).reshape(len(boxes), length)
    # With the count equal, pairwise disjoint leaves cover every sequence.
    for i in range(len(boxes) - 1):
        if np.all((masks[i + 1 :] & masks[i]) != 0, axis=1).any():
            raise CertificateError(f"length {length}: leaves overlap")


def check_certificate(params, spec, forbidden, model_name, cert):
    """Return the accepted answer, or raise CertificateError.

    ``params`` are the float64 numpy weights; UNKNOWN makes no claim and is
    returned unchecked.
    """
    if not isinstance(cert, dict) or cert.get("format") != FORMAT:
        raise CertificateError(f"certificate must be a JSON object with format {FORMAT}")
    if cert.get("model_id") != model_name or cert.get("forbidden") != forbidden:
        raise CertificateError("certificate is for a different model or forbidden token")
    answer = cert.get("answer")
    if answer not in ANSWERS:
        raise CertificateError(f"answer must be one of {', '.join(ANSWERS)}")
    if answer == "COUNTEREXAMPLE":
        sequence = cert.get("sequence")
        box = _box([[token] for token in sequence] if isinstance(sequence, list) else None, spec)
        if box_status(params, spec, forbidden, box)[0] != "violated":
            raise CertificateError(f"{sequence} is not a strict counterexample")
    if answer == "PROVED":
        leaves = cert.get("leaves")
        if not isinstance(leaves, list) or not 0 < len(leaves) <= MAX_LEAVES:
            raise CertificateError(f"leaves must be a list of 1..{MAX_LEAVES} entries")
        by_length = {length: [] for length in range(spec.max_len + 1)}
        for leaf in leaves:
            if not isinstance(leaf, dict):
                raise CertificateError("each leaf must be an object")
            box, witness = _box(leaf.get("box"), spec), _token(leaf.get("witness"), spec)
            if witness == forbidden:
                raise CertificateError("a leaf witness must differ from the forbidden token")
            if box_status(params, spec, forbidden, box, witness)[0] != "proved":
                raise CertificateError(f"leaf {box} with witness {witness} is not proved")
            by_length[len(box)].append(box)
        for length, boxes in by_length.items():
            _check_partition(boxes, spec, length)
    return answer
