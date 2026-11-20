"""Run a submitted verifier once over 100 fresh instances and grade every answer.

Protocol: ``python verifier.py <instances_dir> <answers_dir>``. Each
``<instances_dir>/<id>/`` holds ``config.json`` and ``weights.npz``; the
verifier writes ``<answers_dir>/<id>.json``. One wall-clock limit covers the
whole batch. When it expires the process is killed and the answers already
written still count. A missing or unreadable answer is unsolved; a claim the
checker rejects is wrong.
"""

import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from minimal_nn_competition.verification.certificate import CertificateError, check_answer
from minimal_nn_competition.verification.instances import COUNT, make_instances
from minimal_nn_competition.verification.model import load_instance, save_instance

TIME_LIMIT = 300.0
logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    solved: int = 0
    wrong: int = 0
    unsolved: int = 0
    timed_out: bool = False
    verifier_seconds: float = 0.0
    generation_seconds: float = 0.0
    failures: list = field(default_factory=list)  # (instance id, reason)


def grade(instance_dir, sound, answer_path):
    """Return ``("solved" | "wrong" | "unsolved", reason)`` for one instance."""
    try:
        answer = json.loads(answer_path.read_text())
    except (OSError, ValueError):
        return "unsolved", "no readable answer"
    spec, params, instance_id, forbidden, tau = load_instance(instance_dir)
    try:
        kind = check_answer(params, spec, forbidden, tau, instance_id, answer)
    except CertificateError as exc:
        return "wrong", f"rejected: {exc}"
    if kind == "UNKNOWN":
        return "unsolved", "UNKNOWN"
    # Defence in depth: an accepted claim must also match the construction.
    if (kind == "PROVED") != sound:
        return "wrong", f"{kind} contradicts the ground truth"
    return "solved", kind


def run_evaluation(verifier, spec, seed, time_limit=TIME_LIMIT, count=COUNT):
    """Generate ``count`` instances, run ``verifier`` once and grade the answers."""
    result = Evaluation()
    with tempfile.TemporaryDirectory(prefix="mnnc-verify-") as temp:
        instances_dir, answers_dir = Path(temp) / "instances", Path(temp) / "answers"
        answers_dir.mkdir(parents=True)
        start = time.monotonic()
        truth = {}  # id -> sound; weights stay on disk so memory holds one network at a time
        for instance in make_instances(spec, seed, count):
            directory = instances_dir / instance.instance_id
            save_instance(directory, spec, instance.params, instance.instance_id, 1, instance.tau)
            truth[instance.instance_id] = instance.sound
        result.generation_seconds = time.monotonic() - start
        start = time.monotonic()
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(verifier).resolve()),
                    str(instances_dir),
                    str(answers_dir),
                ],
                cwd=Path(verifier).resolve().parent,
                timeout=time_limit,
                capture_output=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result.timed_out = True
        result.verifier_seconds = time.monotonic() - start
        for instance_id, sound in truth.items():
            answer_path = answers_dir / f"{instance_id}.json"
            outcome, reason = grade(instances_dir / instance_id, sound, answer_path)
            setattr(result, outcome, getattr(result, outcome) + 1)
            if outcome != "solved":
                result.failures.append((instance_id, reason))
                logger.info("%s %s: %s", instance_id, outcome, reason)
    logger.info(
        "solved %d/%d wrong %d unsolved %d verifier %.1fs%s generation %.1fs",
        result.solved,
        count,
        result.wrong,
        result.unsolved,
        result.verifier_seconds,
        " (time limit)" if result.timed_out else "",
        result.generation_seconds,
    )
    return result
