"""The verification transformer's shape, parameter count and instance files.

Tokens are ``0..vocab-1``; token ``vocab`` is BOS and is always at position 0,
so the empty sequence is valid. Each block applies causal multi-head attention
and a ReLU MLP, both added back to the residual stream. There is no LayerNorm
and no unembedding bias. The next token is the argmax of the last position's
logits.

This module imports nothing heavy: the submission check runs it with only
pydantic installed.
"""

import json
import math
from dataclasses import asdict, dataclass

MAX_VOCAB = 64


@dataclass(frozen=True)
class ModelSpec:
    vocab: int
    max_len: int
    d_model: int
    layers: int
    heads: int
    d_mlp: int

    def __post_init__(self):
        if not 2 <= self.vocab <= MAX_VOCAB:
            raise ValueError(f"vocab must be between 2 and {MAX_VOCAB}")
        if min(self.max_len, self.layers) < 0 or min(self.d_model, self.heads, self.d_mlp) < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")

    @property
    def scale(self):
        return 1.0 / math.sqrt(self.d_model // self.heads)

    def shapes(self):
        d, m = self.d_model, self.d_mlp
        shapes: dict[str, tuple[int, ...]] = {
            "embed": (self.vocab + 1, d),
            "pos": (self.max_len + 1, d),
        }
        for layer in range(self.layers):
            prefix = f"layers.{layer}"
            for name in ("wq", "wk", "wv", "wo"):
                shapes[f"{prefix}.{name}"] = (d, d)
            shapes |= {f"{prefix}.w1": (d, m), f"{prefix}.b1": (m,)}
            shapes |= {f"{prefix}.w2": (m, d), f"{prefix}.b2": (d,)}
        return shapes | {"unembed": (d, self.vocab)}

    def parameter_count(self):
        return sum(math.prod(shape) for shape in self.shapes().values())


def save_instance(directory, spec, params, instance_id, forbidden):
    """Write ``config.json`` and float32 ``weights.npz`` for one verification instance."""
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    config = {"id": instance_id, "forbidden": forbidden} | asdict(spec)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    weights = {name: np.asarray(value, dtype=np.float32) for name, value in params.items()}
    np.savez(directory / "weights.npz", **weights)  # ty: ignore[invalid-argument-type]


def load_instance(directory):
    """Return ``(spec, params, instance_id, forbidden)`` with float64 params (exact for float32)."""
    import numpy as np

    config = json.loads((directory / "config.json").read_text())
    instance_id, forbidden = config.pop("id"), config.pop("forbidden")
    with np.load(directory / "weights.npz") as data:
        params = {name: data[name].astype(np.float64) for name in data.files}
    return ModelSpec(**config), params, instance_id, forbidden
