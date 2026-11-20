"""A tiny decoder-only transformer whose next-token behaviour can be verified.

Tokens are ``0..vocab-1``; token ``vocab`` is BOS and is always at position 0,
so the empty sequence is valid. Each block applies causal multi-head attention
and a ReLU MLP, both added back to the residual stream. There is no LayerNorm
and no unembedding bias. The model's meaning is the exact real-valued forward
pass with the stored float32 weights and the float64 attention ``scale``.
"""

import json
import math
from dataclasses import asdict, dataclass

ARCHITECTURE = "verification-transformer"
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


def hidden(params, spec, tokens):
    """Residual stream for a batch of BOS-prefixed token tensors ``(batch, time)``."""
    import torch

    batch, length = tokens.shape
    size = spec.d_model // spec.heads
    x = params["embed"][tokens] + params["pos"][:length]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    for layer in range(spec.layers):
        prefix = f"layers.{layer}"

        def split(name, x=x, prefix=prefix):
            out = x @ params[f"{prefix}.{name}"]
            return out.view(batch, length, spec.heads, size).transpose(1, 2)

        scores = (split("wq") @ split("wk").transpose(-1, -2)) * spec.scale
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        mixed = (weights @ split("wv")).transpose(1, 2).reshape(batch, length, spec.d_model)
        x = x + mixed @ params[f"{prefix}.wo"]
        inner = torch.relu(x @ params[f"{prefix}.w1"] + params[f"{prefix}.b1"])
        x = x + inner @ params[f"{prefix}.w2"] + params[f"{prefix}.b2"]
    return x


def logits(params, spec, tokens):
    return hidden(params, spec, tokens) @ params["unembed"]


def to_numpy(params):
    """float64 copies of the weights for the interval engine (exact for float32)."""
    return {name: value.detach().double().numpy() for name, value in params.items()}


def save_model(directory, spec, params, model_id, forbidden):
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    config = {"architecture": ARCHITECTURE, "model_id": model_id, "forbidden": forbidden}
    (directory / "config.json").write_text(json.dumps(config | asdict(spec), indent=2) + "\n")
    state = {name: value.detach().float().contiguous() for name, value in params.items()}
    torch.save(state, directory / "model.pt")


def load_model(directory):
    """Return ``(spec, params, model_id, forbidden)``; ``params`` are float32 tensors."""
    import torch

    config = json.loads((directory / "config.json").read_text())
    if config.pop("architecture", None) != ARCHITECTURE:
        raise ValueError(f"config architecture must be {ARCHITECTURE!r}")
    model_id = config.pop("model_id")
    forbidden = config.pop("forbidden")
    spec = ModelSpec(**config)
    if not 0 <= forbidden < spec.vocab:
        raise ValueError("forbidden must be a token in the vocabulary")
    params = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    shapes = spec.shapes()
    if set(params) != set(shapes):
        raise ValueError("checkpoint keys do not match the config")
    for name, shape in shapes.items():
        value = params[name]
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    return spec, params, model_id, forbidden
