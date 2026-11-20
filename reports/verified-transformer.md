# Verification: the largest verified transformer

Question: how large can a transformer be while a verifier still decides, within
a time limit, whether a chosen token can ever be its next token?

Code: `src/minimal_nn_competition/verification/` and
`src/minimal_nn_competition/benchmarks/verification.py`. Guide:
`how-to/how-to-make-a-verification-submission.md`.

## Benchmark

- **What is verified:** the verifier is the competitor. A submission claims a
  model shape (`d_model`, `layers`, `heads` and `d_mlp`); the parameter count
  follows from it.
- **Networks:** evaluation builds 100 fresh networks of that shape from a
  random seed. Of these, 50 are sound and 50 hide 1 to 3 inputs where
  forbidden token 1 is the argmax.
- **Run:** the submitted `verifier.py` runs once over all 100 networks, with a
  5-minute limit in total. It passes only when all 100 answers are correct;
  larger models rank first.
- **Inputs:** vocabulary {0, 1}, with BOS at position 0 followed by exactly L
  tokens. There are three length categories:

| Benchmark | L | Inputs per network | Status |
| --- | --- | --- | --- |
| `verify-transformer-len0` | 0 | 1 | Trivial warm-up |
| `verify-transformer-len4` | 4 | 16 | Main tier |
| (planned) | 16 | 65,536 | Work in progress |

## How the networks are built

Generation is numpy only and deterministic from (shape, seed). No training is
needed, so any size can be generated.

1. **Random model:** a random, GPT-2-scaled transformer.
2. **Projection:** every input's final residual `h(s)` is projected onto a
   random direction `w`.
3. **Threshold:** a threshold sits just below the top 1 to 3 projections
   (unsafe) or just above the top one (sound "near miss"). The nearest input
   is the same distance away in both cases, so the two classes look alike.
4. **Planting:** `logit_1 - logit_0 = alpha * (h . w - threshold)` is planted
   exactly, through the forbidden token's unembedding column and the last
   layer's output bias. A random bias component orthogonal to `w` hides the
   construction.
5. **Labels:** every input is at least `tau` from a tie. Labels are
   recomputed after rounding weights to float32, and the build is retried if
   anything moved.

Instance ids are a shuffled position, so they reveal nothing about the label.
Only one network is held in memory at a time.

## Checking answers

Answers are data. The checker accepts these claims:

- **PROVED:** leaves that partition all inputs, each of which is either
  - an `interval` leaf (rigorous interval bounds over a set of sequences), or
  - a `point` leaf (one sequence, with a float64 gap of at least `tau / 2`);
- **COUNTEREXAMPLE:** one sequence where the float64 forward pass puts token 1
  ahead by at least `tau / 2`;
- **UNKNOWN.**

A rejected claim is wrong. The construction guarantees a margin of at least
`tau` on every input, far above float64 rounding error.

## What the field uses

- **CROWN bounds:** alpha-beta-CROWN (built on auto_LiRPA) won VNN-COMP
  2021-2025. It uses linear relaxation bounds (CROWN) plus branch-and-bound,
  and supports transformers.
- **Attention:** softmax and attention bounds are the known bottleneck.
- **SMT and MILP:** these tools (for example Marabou) must approximate
  softmax and scale worst.
- **Our interval bounds:** they are sound but too loose to prove sets of
  sequences. Over the four single-token inputs of one model, the true logit
  gap was 4.8 to 9.5, but the bound was -35 to +84. Interval boxes forget
  that every coordinate comes from the same token. Relational bounds
  (zonotopes or CROWN) are the next step for a stronger verifier.

## Example verifiers

| Benchmark | Example | Method | Claimed size |
| --- | --- | --- | --- |
| len0 | `example-interval` | interval branch-and-bound, point checks at single sequences | 236 parameters |
| len4 | `example-enumerate` | one batched forward pass over all 16 inputs, point leaves | see below |

SCALING_RESULTS
