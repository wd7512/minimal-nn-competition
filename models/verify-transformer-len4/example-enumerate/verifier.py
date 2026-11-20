"""Example verifier: one batched forward pass over every input of each network.

Run as ``python verifier.py <instances_dir> <answers_dir>``.
"""

import json
import sys
from pathlib import Path

import numpy as np

from minimal_nn_competition.verification.forward import all_sequences, gaps, logits
from minimal_nn_competition.verification.model import load_instance


def main(instances, answers):
    for directory in sorted(Path(instances).iterdir()):
        spec, params, instance_id, forbidden = load_instance(directory)
        seqs = all_sequences(spec)
        gap = gaps(logits(params, spec, seqs), forbidden)
        answer = {"id": instance_id, "answer": "PROVED"}
        if gap.max() > 0:
            answer |= {"answer": "COUNTEREXAMPLE", "sequence": list(seqs[int(np.argmax(gap))])}
        # Write then rename, so a kill at the time limit never leaves half a file.
        partial = Path(answers) / f"{instance_id}.json.partial"
        partial.write_text(json.dumps(answer))
        partial.rename(Path(answers) / f"{instance_id}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
