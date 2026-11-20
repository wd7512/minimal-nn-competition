"""Command handlers for ``make-instances`` and ``check-answer``."""

import json
import logging

from minimal_nn_competition.benchmarks.registry import VERIFY_SPECS
from minimal_nn_competition.verification.certificate import CertificateError, check_answer
from minimal_nn_competition.verification.instances import make_instances
from minimal_nn_competition.verification.model import ModelSpec, load_instance, save_instance

logger = logging.getLogger(__name__)


def _make(args):
    spec = ModelSpec(
        **VERIFY_SPECS[args.benchmark],
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        d_mlp=args.d_mlp,
    )
    labels = {}
    for instance in make_instances(spec, args.seed, args.count):
        directory = args.output / "instances" / instance.instance_id
        save_instance(directory, spec, instance.params, instance.instance_id, 1, instance.tau)
        labels[instance.instance_id] = {"sound": instance.sound, "violations": instance.violations}
    # Ground truth for practice only; a real evaluation never shows it to the verifier.
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    logger.info(
        "wrote %d instances of %d parameters to %s",
        len(labels),
        spec.parameter_count(),
        args.output,
    )
    return 0


def _check(args):
    spec, params, instance_id, forbidden, tau = load_instance(args.instance_dir)
    try:
        kind = check_answer(
            params, spec, forbidden, tau, instance_id, json.loads(args.answer.read_text())
        )
    except CertificateError as exc:
        logger.error("REJECTED: %s", exc)
        return 1
    logger.info("ACCEPTED: %s for instance %s", kind, instance_id)
    return 0


def run(args):
    return _make(args) if args.command == "make-instances" else _check(args)
