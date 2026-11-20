# Experimental: largest verified transformer

Question: how large can a transformer be while we can still formally prove
that it never emits a chosen token next (greedy argmax), for every input up to
a maximum length?

Code: `src/minimal_nn_competition/verification/`. Tests:
`tests/test_verification.py`.

## Seed case

Vocabulary {0, 1}, maximum length 2, forbidden token 1. The inputs are `[]`,
`0`, `1`, `00`, `01`, `10`, `11`; a BOS token is prepended so `[]` is valid.

```bash
uv run mnnc verify-transformer --vocab 2 --max-len 2 --d-model 4 --layers 1 --heads 1
```

Result: PROVED for all 7 inputs (244 parameters, 7 certificate leaves, about
0.01 s). A planted variant (`--planted`) returns a checked COUNTEREXAMPLE.

## What the field uses

- alpha-beta-CROWN (built on auto_LiRPA) won VNN-COMP 2021-2025 and ranked
  first on every scored benchmark in 2025. It uses linear relaxation bounds
  (CROWN) plus branch-and-bound, and supports transformers.
- Softmax and attention bounds are the known bottleneck. Recent work such as
  Vertex-Softmax (2026) tightens only the softmax sub-problem.
- SMT and MILP tools (for example Marabou) are complete for ReLU networks but
  must approximate exp and softmax, and scale worst on transformers.
- Discrete token inputs make branch-and-bound over token sets complete:
  splitting down to single sequences is exhaustive enumeration, which is a
  proof when floating-point rounding is bounded soundly.

## Design

- Model: decoder-only transformer, learned token and position embeddings,
  causal multi-head attention and a ReLU MLP per block, residual connections,
  no LayerNorm, no unembedding bias.
- Semantics: the exact real-valued forward pass with the stored float32
  weights. Interval arithmetic in float64 widens every sum by an a-priori
  rounding bound and steps every other result one ulp outward.
- Property on a box (a set of allowed tokens per position): some token j other
  than the forbidden token f has `lo(logit_j) > hi(logit_f)` on the whole box.
- Reference verifier: for each length, start from the box of all sequences
  and split the widest position until every box is proved, a single sequence
  strictly prefers f (counterexample), or a near tie or timeout gives UNKNOWN.

## Evaluating a verifier

The verifier is the competitor. Each instance is a published reference model
plus a forbidden token. The answer is one of:

```json
{"format": 1, "model_id": "...", "forbidden": 1, "answer": "PROVED",
 "leaves": [{"box": [[0, 1], [0]], "witness": 0}]}
{"format": 1, "model_id": "...", "forbidden": 1, "answer": "COUNTEREXAMPLE",
 "sequence": [0, 1]}
{"format": 1, "model_id": "...", "forbidden": 1, "answer": "UNKNOWN"}
```

- PROVED is accepted only when, for every length, the leaves partition all
  sequences exactly (count plus pairwise disjointness) and every leaf passes
  the interval check with its witness. At most 10,000 leaves.
- COUNTEREXAMPLE is accepted only when one interval forward pass shows f is
  strictly the largest logit.
- UNKNOWN scores zero. A rejected claim means the verifier is unsound.
- Oracle files (`--oracle`) give ground truth by checking every input, for
  input spaces up to 100,000 sequences. They test the checker and label
  planted instances; evaluation does not need them.
- Planned headline: the largest parameter count solved per (vocabulary,
  length) tier, ties broken by instances solved, then time.

## First scaling grid

Reference models with seed 0, forbidden token 1, 30 s timeout, heads 2 for
d_model above 4. Measured on the cloud development container.

| V | L | inputs | d_model | layers | params | result | leaves | seconds |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 7 | 4 | 1 | 244 | PROVED | 7 | 0.01 |
| 2 | 2 | 7 | 16 | 2 | 6432 | PROVED | 7 | 0.04 |
| 2 | 2 | 7 | 64 | 2 | 99456 | PROVED | 7 | 0.18 |
| 2 | 2 | 7 | 128 | 4 | 790016 | UNKNOWN (near tie on [0]) | 0 | 0.15 |
| 2 | 4 | 31 | 4 | 1 | 252 | PROVED | 31 | 0.06 |
| 2 | 4 | 31 | 16 | 2 | 6464 | PROVED | 31 | 0.28 |
| 2 | 4 | 31 | 64 | 2 | 99584 | PROVED | 31 | 1.02 |
| 2 | 4 | 31 | 128 | 4 | 790272 | UNKNOWN (near tie on [0]) | 0 | 0.14 |
| 2 | 8 | 511 | 4 | 1 | 268 | PROVED | 511 | 1.57 |
| 2 | 8 | 511 | 16 | 2 | 6528 | COUNTEREXAMPLE | 0 | 0.01 |
| 2 | 8 | 511 | 64 | 2 | 99840 | UNKNOWN (timeout) | 0 | 30.01 |
| 2 | 8 | 511 | 128 | 4 | 790784 | UNKNOWN (near tie on [0]) | 0 | 0.13 |
| 4 | 4 | 341 | 4 | 1 | 268 | PROVED | 341 | 0.71 |
| 4 | 4 | 341 | 16 | 2 | 6528 | PROVED | 341 | 3.13 |
| 4 | 4 | 341 | 64 | 2 | 99840 | PROVED | 341 | 12.15 |
| 4 | 4 | 341 | 128 | 4 | 790784 | UNKNOWN (near tie on [1, 1, 1, 1]) | 0 | 21.09 |

Findings:

- Plain interval bounds never proved a set of more than one sequence: every
  PROVED certificate has one leaf per input, so this is sound enumeration.
  Larger input spaces need tighter bounds over token sets.
- Even on single sequences, interval widths grow by roughly 10^3 to 10^4 per
  attention layer. At 4 layers and d_model 128 the bounds no longer separate
  the logits (UNKNOWN, near tie) although the model is safe on those inputs.
- COUNTEREXAMPLE rows are real: training stopped before the model preferred
  token 0 on every input.

## Next steps

1. Grid script and report for vocabulary up to 16 and length up to 8.
2. CROWN-style linear bounds and direct bounds on `logit_j - logit_f`, so set
   boxes can be proved and deeper models stay tight. The checker re-runs them.
3. Export reference models and specs to ONNX and VNN-LIB, and run
   alpha-beta-CROWN and Marabou as external baselines.
4. Competition wiring: published reference grid, certificate submissions,
   and a larger-is-better leaderboard.
