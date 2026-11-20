"""Deterministic verification instances: 50 sound networks and 50 with sparse counterexamples.

Every instance comes from ``(spec, seed, index)`` with numpy's PCG64 generator
and needs no training, so any size can be generated and the same seed gives
the same networks on every platform.

Construction (binary vocabulary, forbidden token 1, at least one layer):

1. Draw a well-scaled random transformer and compute every input's final
   residual ``h(s)`` with a float64 forward pass.
2. Project onto a random direction ``w``: ``p(s) = h(s) . w``. Pick a
   threshold ``theta`` between the k-th and (k+1)-th largest projections
   (k = 1..3 violations) or above the largest (a sound near miss, as far
   from the threshold as an unsafe network's nearest point).
3. Make ``logit_1 - logit_0 = alpha * (p(s) - theta)`` exactly: set
   ``u_1 = u_0 + alpha * w`` and add ``-theta * w / |w|^2`` to the last layer's
   output bias, which only shifts the final residual. A random component
   orthogonal to ``w`` is added to that bias to hide the construction.
4. ``alpha`` puts the closest input exactly ``2 * tau`` from the threshold.
   After rounding the weights to float32, the labels are recomputed and the
   instance is redrawn if any input is within ``tau`` of a tie.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from minimal_nn_competition.verification.forward import all_sequences, gaps, residuals

COUNT = 100
TAU = 0.5  # every input's logit gap is at least TAU from a tie
MAX_VIOLATIONS = 3
MAX_ATTEMPTS = 50


@dataclass
class Instance:
    instance_id: str
    params: dict
    violations: list = field(default_factory=list)  # ground truth; empty means sound


def random_params(spec, rng):
    """GPT-2-style init: embeddings N(0, 1), matrices N(0, 1/fan_in), zero biases."""
    params = {}
    for name, shape in spec.shapes().items():
        if len(shape) == 1:
            params[name] = np.zeros(shape)
            continue
        std = 1.0 if name in ("embed", "pos") else 1 / math.sqrt(shape[0])
        if name.endswith((".wo", ".w2")):
            std /= math.sqrt(2 * spec.layers)
        params[name] = rng.normal(0.0, std, size=shape)
    return _float32(params)


def _float32(params):
    return {name: value.astype(np.float32).astype(np.float64) for name, value in params.items()}


def _threshold(p, sound, rng):
    """Threshold and the input count above it; sound nets get a near miss above the top."""
    top = np.sort(p)[::-1]
    if len(top) == 1:
        return (top[0] + 1.0, 0) if sound else (top[0] - 1.0, 1)
    if sound:
        return top[0] + (top[0] - top[1]) / 2, 0
    k = int(rng.integers(1, min(MAX_VIOLATIONS, len(top) - 1) + 1))
    return (top[k - 1] + top[k]) / 2, k


def _shape(params, spec, seqs, rng, sound):
    """Try once to plant the property; return ``(params, violations)`` or None."""
    h = residuals(params, spec, seqs)
    w = rng.normal(size=spec.d_model)
    p = h @ w
    theta, k = _threshold(p, sound, rng)
    closest = float(np.min(np.abs(p - theta)))
    if closest <= 0:
        return None
    alpha = 2 * TAU / closest
    last = f"layers.{spec.layers - 1}.b2"
    hidden = rng.normal(size=spec.d_model)
    hidden -= (hidden @ w) / (w @ w) * w  # orthogonal to w: does not change any gap
    unembed = params["unembed"].copy()
    unembed[:, 1] = unembed[:, 0] + alpha * w
    shaped = _float32(params | {"unembed": unembed, last: -theta * w / (w @ w) + hidden})
    gap = gaps(residuals(shaped, spec, seqs) @ shaped["unembed"], 1)
    violations = [list(seqs[i]) for i in np.flatnonzero(gap > 0)]
    if np.any(np.abs(gap) < TAU) or len(violations) != k:
        return None
    return shaped, violations


def make_instance(spec, seed, index, sound=True):
    """Build one labelled instance; raises if no candidate fits within MAX_ATTEMPTS."""
    if spec.vocab != 2 or spec.layers < 1:
        raise ValueError("instances need vocab 2 and at least one layer")
    rng = np.random.default_rng([seed, index])
    seqs = all_sequences(spec)
    for _ in range(MAX_ATTEMPTS):
        shaped = _shape(random_params(spec, rng), spec, seqs, rng, sound)
        if shaped is not None:
            return Instance(f"{index:03d}", shaped[0], shaped[1])
    raise RuntimeError(f"could not build instance {seed}-{index} for {spec}")


def make_instances(spec, seed, count=COUNT):
    """Yield half sound, half sparse-counterexample instances in a seeded random order.

    A generator, so only one network is in memory at a time.
    """
    order = np.random.default_rng([seed, count]).permutation(count)
    for position, i in enumerate(order):
        instance = make_instance(spec, seed, int(i), sound=bool(i < count // 2))
        instance.instance_id = f"{position:03d}"  # opaque: reveals nothing about the label
        yield instance
