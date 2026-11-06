# Proof certificates

This directory contains the machine-checked part of the parameter-bound
research. It is deliberately outside the Python package: the competition
runtime and submission rules do not depend on Lean.

## What is proved

For CartPole (`input_size = 4`, `output_size = 2`), every accepted MLP
architecture with positive hidden widths has at least nine parameters:

```
MinimalNN.cartPole_architecturalMinimum
  : IsArchitecturalMinimum 4 2 9
```

The theorem is in [`lean/Proofs/CartPole.lean`](lean/Proofs/CartPole.lean) and
builds on the generic count theorem in
[`lean/Proofs/Architecture.lean`](lean/Proofs/Architecture.lean):

```
MinimalNN.parameterCount_ge_floor
  : architecturalFloor d k <= parameterCount d k hidden
```

**Scope.** The Lean theorem is about parameter arithmetic and the declared
architecture family. It does not formalise Gymnasium, Box2D, float32 inference,
or the pass condition. The 9-parameter model passing all 100 seeds is evidence
from the benchmark evaluator, not from Lean. The combination is recorded in
[`bounds.json`](bounds.json) and labelled `combined`.

## Evidence classes

| Class | Establishes | Checked by |
| --- | --- | --- |
| Formal theorem | every accepted architecture has at least 9 parameters | Lean kernel |
| Computation certificate | generated parameter/floor claims match the pinned table | `decide` during `lake build` |
| Empirical witness | `relu-threshold-9` passes 100/100 seeds | benchmark evaluator |
| Future solver certificate | infeasibility or observability certificate | Lean checker (future PR) |

[`lean/Proofs/Generated.lean`](lean/Proofs/Generated.lean) is the checkable
computation certificate. [`bounds.json`](bounds.json) is a manifest that names
the theorem, records the evaluator result, and states which evidence classes the
combined conclusion requires.

## Layout

```
proofs/
  architectures.json       pinned parameter-count table
  acceptance.json          pinned acceptance contract and source hashes
  bounds.json              generated manifest
  pipeline.py              generation, evaluation and comparison
  lean/                    Lean 4 project, no Mathlib
```

The acceptance contract covers the rules the theorem depends on: fixed
input/output sizes, positive hidden widths, `mlp` only, float32 checkpoints, and
no feature selection for CartPole. Behavioural tests are the primary guard; the
pinned hashes are a tripwire that forces a proof review when those rules change.

## Commands

Build the proofs (Lean 4.35.0, pinned by `lean/lean-toolchain`):

```bash
cd proofs/lean
lake build
```

If Lean is not installed, a workspace-local toolchain works:

```bash
export ELAN_HOME="$PWD/.elan"
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain none
export PATH="$ELAN_HOME/bin:$PATH"
```

Regenerate and check the certificate artifacts:

```bash
uv run python -m proofs generate
uv run python -m proofs check
```

`check` re-evaluates the 100-seed witness, compares both generated artifacts,
and fails if an acceptance source hash changed. The simpler
`python -m proofs check-lean` checks only the generated Lean file and has no
torch dependency; CI uses it before installing Lean.

Run the focused tests:

```bash
uv run pytest tests/test_proofs.py
```

## Regeneration rules

- `architectures.json` and `acceptance.json` are hand-maintained and reviewed.
- `Generated.lean` and `bounds.json` are generated. Do not edit them by hand.
- If `acceptance.json` hashes fail, revalidate the CartPole theorem against the
  new submission rules before updating the hashes.
- If a passing CartPole submission with fewer than nine parameters appears, the
  leaderboard test fails. That is a contradiction to investigate, not a file to
  regenerate.

## Why no Mathlib

The current proof needs only `Nat`, list induction, `omega`, `simp` and
`decide`. Mathlib would add a large dependency for no benefit at this stage.
Add it when a later theorem needs probability, entropy, or real arithmetic.
