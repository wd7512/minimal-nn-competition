"""Decode leader checkpoints to JSON (temporary helper)."""

import json
from pathlib import Path

import torch

OUT = Path("glm-weights.json")
NAMES = [
    "hamiltonian-3-h6",
    "a-hamiltonian-3-h6",
    "serpentine-8-h6",
    "serpentine-6-6-fs14",
    "hamiltonian-8-8-fs14",
]

def main() -> None:
    out = {}
    for bench, names in [
        ("snake-simple-8x8", NAMES),
        ("snake-simple-16x16", ["cycle-16x6", "cycle-16x7"]),
    ]:
        for name in names:
            path = Path("models") / bench / name / "model.pt"
            if not path.is_file():
                out[f"{bench}/{name}"] = {"error": "missing"}
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                out[f"{bench}/{name}"] = {
                    k: v.tolist() for k, v in state.items()
                }
            except Exception as exc:  # noqa: BLE001
                out[f"{bench}/{name}"] = {"error": repr(exc)}
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
