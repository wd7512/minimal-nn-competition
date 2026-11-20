"""Deterministic reference models for the verification grid.

A reference model is trained briefly on a fixed next-token task whose targets
never include the forbidden token. Training does not guarantee the property;
the verifier decides. A planted variant edits the unembedding so the forbidden
token wins on one chosen sequence, giving a known counterexample.
"""

import itertools
import math
import random

from minimal_nn_competition.verification.model import ModelSpec, hidden, logits

MAX_TRAINING_SEQUENCES = 4096
STOP_LOSS = 0.01


def model_id(spec, seed, planted=False):
    name = f"ref-v{spec.vocab}-l{spec.max_len}-d{spec.d_model}-n{spec.layers}-h{spec.heads}"
    return f"{name}-m{spec.d_mlp}-s{seed}" + ("-planted" if planted else "")


def sequences(vocab, length):
    return itertools.product(range(vocab), repeat=length)


def all_sequences(spec):
    for length in range(spec.max_len + 1):
        yield from sequences(spec.vocab, length)


def input_count(spec):
    return sum(spec.vocab**length for length in range(spec.max_len + 1))


def target(sequence, vocab, forbidden):
    allowed = [token for token in range(vocab) if token != forbidden]
    return allowed[(sum(sequence) + len(sequence)) % len(allowed)]


def batch(spec, seqs):
    """BOS-prefixed, right-padded token tensor and each sequence's last index."""
    import torch

    width = max(len(seq) for seq in seqs) + 1
    tokens = torch.zeros(len(seqs), width, dtype=torch.long)
    tokens[:, 0] = spec.vocab
    for row, seq in enumerate(seqs):
        tokens[row, 1 : len(seq) + 1] = torch.tensor(seq, dtype=torch.long)
    return tokens, torch.tensor([len(seq) for seq in seqs])


def next_logits(params, spec, seqs):
    """Next-token logits for each sequence; right padding is ignored by the causal mask."""
    import torch

    tokens, last = batch(spec, seqs)
    return logits(params, spec, tokens)[torch.arange(len(seqs)), last]


def _training_set(spec, forbidden, rng):
    if input_count(spec) <= MAX_TRAINING_SEQUENCES:
        seqs = list(all_sequences(spec))
    else:
        seqs = [
            tuple(rng.randrange(spec.vocab) for _ in range(rng.randint(0, spec.max_len)))
            for _ in range(MAX_TRAINING_SEQUENCES)
        ]
    return seqs, [target(seq, spec.vocab, forbidden) for seq in seqs]


def make_reference(spec, seed, forbidden=1, steps=300, lr=3e-3):
    """Train a reference model; returns float32 parameter tensors."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    params = {}
    for name, shape in spec.shapes().items():
        # Embeddings N(0, 1), matrices N(0, 1/fan_in), zero biases; residual
        # outputs shrink by sqrt(2 * layers) as in GPT-2 to keep activations small.
        std = 1.0 if name in ("embed", "pos") else 1 / math.sqrt(shape[0])
        if name.endswith((".wo", ".w2")):
            std /= math.sqrt(2 * spec.layers)
        value = (
            torch.randn(shape, generator=generator) * std if len(shape) == 2 else torch.zeros(shape)
        )
        params[name] = value.requires_grad_()
    seqs, targets = _training_set(spec, forbidden, random.Random(seed))
    labels = torch.tensor(targets)
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(next_logits(params, spec, seqs), labels)
        # Stop early: without LayerNorm, separable targets otherwise push logits to huge values.
        if loss.item() < STOP_LOSS:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()
    return {name: value.detach() for name, value in params.items()}


def plant_violation(params, spec, forbidden, sequence, margin=1.0):
    """Edit the unembedding so ``forbidden`` beats every other token on ``sequence``."""

    double = {name: value.double() for name, value in params.items()}
    tokens, _ = batch(spec, [sequence])
    h = hidden(double, spec, tokens)[0, -1]
    scores = h @ double["unembed"]
    best_other = max(scores[j] for j in range(spec.vocab) if j != forbidden)
    boost = (best_other - scores[forbidden] + margin) / (h @ h)
    unembed = double["unembed"].clone()
    unembed[:, forbidden] += boost * h
    return params | {"unembed": unembed.float()}


def reference_spec(vocab, max_len, d_model, layers, heads, d_mlp=None):
    return ModelSpec(vocab, max_len, d_model, layers, heads, d_mlp or 4 * d_model)
