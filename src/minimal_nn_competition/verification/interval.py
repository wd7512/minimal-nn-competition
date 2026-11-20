"""Sound interval bounds for the verification transformer.

This module is the trusted core: the certificate checker relies on it. Every
interval ``[lo, hi]`` contains the exact real-valued result for every input in
the box. Weights are float64 values and are treated as exact.

Floating-point rounding is handled two ways. Sums are widened by the a-priori
bound ``gamma(n) * sum(|terms|)`` (Higham, Accuracy and Stability of Numerical
Algorithms, section 3.1), which holds for any summation order. Every other
result is stepped one ulp outward with ``nextafter``.
"""

import numpy as np

UNIT_ROUNDOFF = 2.0**-53
SMALLEST = 2.0**-1074
# numpy's exp is accurate to a few ulps; widen far beyond that.
EXP_SLACK = 2.0**-40


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def _clean(lo, hi):
    """Replace NaN, for example from ``inf - inf``, with an unbounded interval."""
    return np.where(np.isnan(lo), -np.inf, lo), np.where(np.isnan(hi), np.inf, hi)


def add(alo, ahi, blo, bhi):
    return _clean(down(alo + blo), up(ahi + bhi))


def mul(alo, ahi, blo, bhi):
    products = np.stack(np.broadcast_arrays(alo * blo, alo * bhi, ahi * blo, ahi * bhi))
    lo, hi = _clean(products.min(axis=0), products.max(axis=0))
    return down(lo), up(hi)


def total(lo, hi, axis):
    """Sum intervals along ``axis`` with a rigorous rounding-error allowance."""
    n = lo.shape[axis]
    gamma = (n + 1) * UNIT_ROUNDOFF / (1 - (n + 1) * UNIT_ROUNDOFF)
    magnitude = np.maximum(np.abs(lo), np.abs(hi)).sum(axis=axis)
    error = 2 * gamma * magnitude + (n + 1) * SMALLEST
    return _clean(down(lo.sum(axis=axis) - error), up(hi.sum(axis=axis) + error))


def linear(lo, hi, weight, bias=None):
    """Bound ``x @ weight + bias`` for ``x`` in ``[lo, hi]`` (last axis contracted)."""
    plo, phi = mul(lo[..., :, None], hi[..., :, None], weight, weight)
    lo, hi = total(plo, phi, axis=-2)
    if bias is not None:
        lo, hi = add(lo, hi, bias, bias)
    return lo, hi


def relu(lo, hi):
    return np.maximum(lo, 0.0), np.maximum(hi, 0.0)


def exp(lo, hi):
    return np.maximum(down(np.exp(lo) * (1 - EXP_SLACK)), 0.0), up(np.exp(hi) * (1 + EXP_SLACK))


def softmax(lo, hi):
    """Bound softmax over the last axis using p_i = 1 / (1 + sum_{j!=i} exp(s_j - s_i))."""
    n = lo.shape[-1]
    others = ~np.eye(n, dtype=bool)
    # Upper bound on each denominator gives the lower bound on p_i.
    _, gap_hi = add(hi[..., None, :], hi[..., None, :], -lo[..., :, None], -lo[..., :, None])
    _, e_hi = exp(gap_hi, gap_hi)
    e_hi = np.where(others, e_hi, 0.0)
    _, den_hi = total(e_hi, e_hi, axis=-1)
    p_lo = np.maximum(down(1.0 / up(1.0 + den_hi)), 0.0)
    # Lower bound on each denominator gives the upper bound on p_i.
    gap_lo, _ = add(lo[..., None, :], lo[..., None, :], -hi[..., :, None], -hi[..., :, None])
    e_lo, _ = exp(gap_lo, gap_lo)
    e_lo = np.where(others, e_lo, 0.0)
    den_lo, _ = total(e_lo, e_lo, axis=-1)
    p_hi = np.minimum(up(1.0 / down(1.0 + den_lo)), 1.0)
    return p_lo, p_hi


def _attention(lo, hi, params, prefix, heads, scale):
    length, width = lo.shape
    size = width // heads
    q = linear(lo, hi, params[f"{prefix}.wq"])
    k = linear(lo, hi, params[f"{prefix}.wk"])
    v = linear(lo, hi, params[f"{prefix}.wv"])
    out_lo = np.zeros((length, width))
    out_hi = np.zeros((length, width))
    for head in range(heads):
        cols = slice(head * size, (head + 1) * size)
        qh = (q[0][:, cols], q[1][:, cols])
        kh = (k[0][:, cols], k[1][:, cols])
        vh = (v[0][:, cols], v[1][:, cols])
        for t in range(length):
            # Causal: position t attends to positions 0..t.
            s = mul(qh[0][t], qh[1][t], kh[0][: t + 1], kh[1][: t + 1])
            s = total(*s, axis=-1)
            s = mul(s[0], s[1], scale, scale)
            p_lo, p_hi = softmax(*s)
            o = mul(p_lo[:, None], p_hi[:, None], vh[0][: t + 1], vh[1][: t + 1])
            out_lo[t, cols], out_hi[t, cols] = total(*o, axis=0)
    return linear(out_lo, out_hi, params[f"{prefix}.wo"])


def bound_logits(params, spec, box):
    """Bound the next-token logits over every sequence in ``box``.

    ``box`` holds one non-empty collection of tokens per position. The BOS
    token is prepended. Each position's embedding is bounded by the hull of its
    tokens' embeddings, which is sound but forgets that one token is chosen.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return _bound_logits(params, spec, box)


def _bound_logits(params, spec, box):
    tokens = [[spec.vocab]] + [sorted(position) for position in box]
    embed = params["embed"]
    lo = np.stack([embed[group].min(axis=0) for group in tokens])
    hi = np.stack([embed[group].max(axis=0) for group in tokens])
    pos = params["pos"][: len(tokens)]
    lo, hi = add(lo, hi, pos, pos)
    for layer in range(spec.layers):
        prefix = f"layers.{layer}"
        alo, ahi = _attention(lo, hi, params, prefix, spec.heads, spec.scale)
        lo, hi = add(lo, hi, alo, ahi)
        mlo, mhi = linear(lo, hi, params[f"{prefix}.w1"], params[f"{prefix}.b1"])
        mlo, mhi = linear(*relu(mlo, mhi), params[f"{prefix}.w2"], params[f"{prefix}.b2"])
        lo, hi = add(lo, hi, mlo, mhi)
    return linear(lo[-1], hi[-1], params["unembed"])


def box_status(params, spec, forbidden, box, witness=None):
    """Return ``(True, witness)`` if the interval bounds prove the box safe.

    Proved means some token ``j != forbidden`` has ``lo(logit_j) > hi(logit_forbidden)``
    on every sequence in the box. ``witness`` fixes ``j``; otherwise the best
    candidate is tried.
    """
    lo, hi = bound_logits(params, spec, box)
    others = [j for j in range(spec.vocab) if j != forbidden]
    if witness is None:
        witness = max(others, key=lambda j: lo[j])
    return bool(lo[witness] > hi[forbidden]), witness
