"""Example verifier: check every input with one batched forward pass per network.

Finds the strongest counterexample if one exists, otherwise proves the network
with one point leaf per input. Run as ``python verifier.py <instances> <answers>``.
"""

import json
import sys
from pathlib import Path

import numpy as np

from minimal_nn_competition.verification.forward import all_sequences, gaps, logits
from minimal_nn_competition.verification.model import load_instance


def answer_for(params, spec, forbidden, tau):
    seqs = all_sequences(spec)
    rows = logits(params, spec, seqs)
    gap = gaps(rows, forbidden)
    worst = int(np.argmax(gap))
    if gap[worst] >= tau / 2:
        return {"answer": "COUNTEREXAMPLE", "sequence": list(seqs[worst])}
    if np.any(gap > -tau / 2):
        return {"answer": "UNKNOWN"}
    others = np.delete(np.arange(spec.vocab), forbidden)
    witnesses = others[np.argmax(rows[:, others], axis=1)]
    leaves = [
        {"box": [[t] for t in seq], "witness": int(w), "method": "point"}
        for seq, w in zip(seqs, witnesses, strict=True)
    ]
    return {"answer": "PROVED", "leaves": leaves}


def main(instances, answers):
    for directory in sorted(Path(instances).iterdir()):
        spec, params, instance_id, forbidden, tau = load_instance(directory)
        answer = {"id": instance_id} | answer_for(params, spec, forbidden, tau)
        partial = Path(answers) / f"{instance_id}.json.partial"
        partial.write_text(json.dumps(answer))
        partial.rename(Path(answers) / f"{instance_id}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
