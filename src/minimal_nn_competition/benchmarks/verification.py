"""Verification benchmarks: the largest transformer a submitted verifier handles in time.

A submission claims a model shape. Evaluation generates 100 fresh networks of
that shape (50 sound, 50 with 1-3 counterexamples), runs the submitted
``verifier.py`` once over all of them under one time limit, and checks every
answer. It passes only when all 100 are solved; larger models rank higher.
"""

import secrets

from minimal_nn_competition.verification.instances import COUNT
from minimal_nn_competition.verification.runner import TIME_LIMIT, run_evaluation


class VerificationBenchmark:
    name = ""

    @classmethod
    def factory(cls, name):
        return type(f"VerificationBenchmark_{name}", (cls,), {"name": name})

    def __init__(self, seed=None, time_limit=TIME_LIMIT):
        # A fresh seed per evaluation, so verifiers cannot be tuned to known networks.
        self.seed = secrets.randbelow(2**32) if seed is None else seed
        self.time_limit = time_limit

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError(f"Submission is for {submission.config.benchmark}, not {self.name}")
        return submission

    def evaluate(self, submission):
        spec = submission.config.model_spec()
        result = run_evaluation(submission.checkpoint, spec, self.seed, self.time_limit)
        details = {
            "solved": result.solved,
            "wrong": result.wrong,
            "unsolved": result.unsolved,
            "instances": COUNT,
            "inputs_per_network": spec.vocab**spec.max_len,
            "seed": self.seed,
            "time_limit_seconds": self.time_limit,
            "verifier_seconds": round(result.verifier_seconds, 2),
            "generation_seconds": round(result.generation_seconds, 2),
        }
        return result.solved / COUNT, result.solved == COUNT, details
