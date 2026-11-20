"""Run a submitted verifier once over 100 fresh networks and grade every answer.

Protocol: ``python verifier.py <instances_dir> <answers_dir>``. Each
``<instances_dir>/<id>/`` holds ``config.json`` and ``weights.npz``; the
verifier writes ``<answers_dir>/<id>.json`` as ``{"id", "answer": "PROVED"}`` or
``{"id", "answer": "COUNTEREXAMPLE", "sequence": [...]}``. One wall-clock
limit covers the batch; when it expires the process is killed and answers
already written still count.

Answers are graded against the labels planted at construction: PROVED is
right only on a sound network, and a counterexample only if it is one of the
planted violations. A false claim is wrong; anything else is unsolved.
"""

import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from minimal_nn_competition.verification.instances import COUNT, make_instances
from minimal_nn_competition.verification.model import save_instance

TIME_LIMIT = 300.0
logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    solved: int = 0
    wrong: int = 0
    unsolved: int = 0
    verifier_seconds: float = 0.0


def grade(violations, answer_path):
    """Return ``"solved"``, ``"wrong"`` or ``"unsolved"`` for one network."""
    try:
        answer = json.loads(answer_path.read_text())
    except (OSError, ValueError):
        return "unsolved"
    kind = answer.get("answer") if isinstance(answer, dict) else None
    if kind == "PROVED":
        return "wrong" if violations else "solved"
    if kind == "COUNTEREXAMPLE":
        return "solved" if answer.get("sequence") in violations else "wrong"
    return "unsolved"


def run_evaluation(verifier, spec, seed, time_limit=TIME_LIMIT, count=COUNT):
    """Generate ``count`` networks, run ``verifier`` once and grade the answers."""
    result = Evaluation()
    verifier = Path(verifier).resolve()
    with tempfile.TemporaryDirectory(prefix="mnnc-verify-") as temp:
        instances_dir, answers_dir = Path(temp) / "instances", Path(temp) / "answers"
        answers_dir.mkdir(parents=True)
        truth = {}  # id -> planted violations; weights stay on disk, one network in memory
        for instance in make_instances(spec, seed, count):
            save_instance(
                instances_dir / instance.instance_id, spec, instance.params, instance.instance_id, 1
            )
            truth[instance.instance_id] = instance.violations
        start = time.monotonic()
        try:
            command = [sys.executable, str(verifier), str(instances_dir), str(answers_dir)]
            subprocess.run(command, cwd=verifier.parent, timeout=time_limit, capture_output=True)
        except subprocess.TimeoutExpired:
            logger.info("verifier stopped at the %.0fs time limit", time_limit)
        result.verifier_seconds = time.monotonic() - start
        for instance_id, violations in truth.items():
            outcome = grade(violations, answers_dir / f"{instance_id}.json")
            setattr(result, outcome, getattr(result, outcome) + 1)
            if outcome != "solved":
                logger.info("network %s: %s", instance_id, outcome)
    logger.info(
        "solved %d/%d, wrong %d, verifier %.1fs",
        result.solved,
        count,
        result.wrong,
        result.verifier_seconds,
    )
    return result
