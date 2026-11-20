"""Batched float64 forward pass, used for labels, point leaves and counterexamples."""

import itertools

import numpy as np


def sequences(vocab, length):
    return list(itertools.product(range(vocab), repeat=length))


def all_sequences(spec):
    """The input space: every sequence of exactly ``max_len`` tokens (after BOS)."""
    return sequences(spec.vocab, spec.max_len)


def input_count(spec):
    return spec.vocab**spec.max_len


def _softmax(scores):
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def residuals(params, spec, seqs, chunk=256):
    """Final residual at the last position for each sequence, shape ``(n, d_model)``.

    Sequences are BOS-prefixed and right-padded; the causal mask hides the
    padding from earlier positions. ``chunk`` bounds memory for large models.
    """
    outputs = []
    size = spec.d_model // spec.heads
    for start in range(0, len(seqs), chunk):
        batch = seqs[start : start + chunk]
        n, length = len(batch), max(len(seq) for seq in batch) + 1
        # tokens[:, 0] is BOS; sequences may differ in length (padding is harmless).
        tokens = np.zeros((n, length), dtype=np.int64)
        tokens[:, 0] = spec.vocab
        for row, seq in enumerate(batch):
            tokens[row, 1 : len(seq) + 1] = seq
        x = params["embed"][tokens] + params["pos"][:length]
        future = ~np.tri(length, dtype=bool)
        for layer in range(spec.layers):
            p = f"layers.{layer}"
            q, k, v = (
                (x @ params[f"{p}.{name}"])
                .reshape(n, length, spec.heads, size)
                .transpose(0, 2, 1, 3)
                for name in ("wq", "wk", "wv")
            )
            scores = np.where(future, -np.inf, (q @ k.transpose(0, 1, 3, 2)) * spec.scale)
            mixed = (_softmax(scores) @ v).transpose(0, 2, 1, 3).reshape(n, length, -1)
            x = x + mixed @ params[f"{p}.wo"]
            x = x + np.maximum(x @ params[f"{p}.w1"] + params[f"{p}.b1"], 0.0) @ params[f"{p}.w2"]
            x = x + params[f"{p}.b2"]
        outputs.append(x[np.arange(n), [len(seq) for seq in batch]])
    return np.concatenate(outputs)


def logits(params, spec, seqs):
    return residuals(params, spec, seqs) @ params["unembed"]


def gaps(logit_rows, forbidden):
    """``logit_forbidden - max_{j != forbidden} logit_j`` per row; positive means violated."""
    others = np.delete(logit_rows, forbidden, axis=1)
    return logit_rows[:, forbidden] - others.max(axis=1)
