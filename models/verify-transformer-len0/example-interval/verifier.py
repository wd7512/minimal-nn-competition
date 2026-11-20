"""Example verifier: branch-and-bound with interval bounds, settling single sequences
with the float64 point check. Run as ``python verifier.py <instances> <answers>``."""

import json
import sys
from pathlib import Path

from minimal_nn_competition.verification.model import load_instance
from minimal_nn_competition.verification.verify import branch_and_bound


def main(instances, answers):
    for directory in sorted(Path(instances).iterdir()):
        spec, params, instance_id, forbidden, tau = load_instance(directory)
        answer = {"id": instance_id} | branch_and_bound(params, spec, forbidden, tau)
        # Write then rename, so a kill at the time limit never leaves half a file.
        partial = Path(answers) / f"{instance_id}.json.partial"
        partial.write_text(json.dumps(answer))
        partial.rename(Path(answers) / f"{instance_id}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
