"""Command handlers for ``verify-transformer`` and ``check-certificate``."""

import json
import logging
import math

from minimal_nn_competition.verification.certificate import (
    CertificateError,
    check_certificate,
    make_certificate,
)
from minimal_nn_competition.verification.model import load_model, save_model, to_numpy
from minimal_nn_competition.verification.reference import (
    input_count,
    make_reference,
    model_id,
    plant_violation,
    reference_spec,
)
from minimal_nn_competition.verification.verify import branch_and_bound, oracle

logger = logging.getLogger(__name__)
PLANTED_SEQUENCE = (0, 1)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("wrote %s", path)


def _size(box):
    return math.prod(len(position) for position in box)


def _verify(args):
    spec = reference_spec(args.vocab, args.max_len, args.d_model, args.layers, args.heads)
    if not 0 <= args.forbidden < spec.vocab:
        raise ValueError("forbidden must be a token in the vocabulary")
    name = model_id(spec, args.seed, args.planted)
    params = make_reference(spec, args.seed, args.forbidden)
    if args.planted:
        params = plant_violation(params, spec, args.forbidden, PLANTED_SEQUENCE[: spec.max_len])
    if args.save_model:
        save_model(args.save_model, spec, params, name, args.forbidden)
    weights = to_numpy(params)
    result = branch_and_bound(weights, spec, args.forbidden, timeout=args.timeout)
    logger.info(
        "%s params=%d %s %s inputs=%d/%d covered leaves=%d boxes=%d %.3fs",
        name,
        spec.parameter_count(),
        result.status,
        result.counterexample if result.counterexample is not None else result.reason,
        sum(_size(box) for box, _ in result.leaves),
        input_count(spec),
        len(result.leaves),
        result.boxes,
        result.seconds,
    )
    if args.cert:
        _write(args.cert, make_certificate(result, name, args.forbidden))
    if args.oracle:
        _write(args.oracle, oracle(weights, spec, args.forbidden, name))
    return 0


def _check(args):
    spec, params, name, forbidden = load_model(args.model_dir)
    cert = json.loads(args.certificate.read_text())
    try:
        answer = check_certificate(to_numpy(params), spec, forbidden, name, cert)
    except CertificateError as exc:
        logger.error("REJECTED: %s", exc)
        return 1
    logger.info("ACCEPTED: %s for %s", answer, name)
    return 0


def run(args):
    return _verify(args) if args.command == "verify-transformer" else _check(args)
