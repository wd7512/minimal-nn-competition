# How to make a verification submission

Verification turns the competition around: you submit a *verifier* and win by
handling the *largest* transformer. Your config claims a model shape.
Evaluation builds 100 fresh random networks of that shape:

- 50 are sound: token 1 is never the next token;
- 50 hide 1 to 3 inputs where token 1 is the next token.

Your verifier runs once over all 100, with a 5-minute limit in total. You pass
only when every answer is right. Larger models rank first.

| Benchmark | Inputs per network |
| --- | --- |
| `verify-transformer-len0` | 1 (just the BOS token) |
| `verify-transformer-len4` | 16 (every 4-token binary sequence) |

A length-16 tier (65,536 inputs) is work in progress.

## The networks

- **Inputs:** vocabulary `{0, 1}`, with BOS (token 2) at position 0 followed
  by exactly 0 or 4 tokens.
- **Architecture:** a decoder-only transformer with no LayerNorm:
  - learned token and position embeddings;
  - `layers` blocks, each with causal multi-head attention and a ReLU MLP,
    added back to the residual stream;
  - an `unembed` matrix with no bias.
- **Prediction:** the argmax of the last position's logits.
- **Reference code:** `verification/forward.py` is the reference forward pass,
  and `verification/model.py` lists every weight's name and shape.
- **Margin:** every input's logit gap is at least 0.5 from a tie.

## Files

Open a pull request against `submission` with exactly two files:

```text
models/verify-transformer-len4/<name>/config.json
models/verify-transformer-len4/<name>/verifier.py
```

```json
{
  "benchmark": "verify-transformer-len4",
  "name": "example",
  "author": "example author",
  "architecture": "verifier",
  "d_model": 64,
  "layers": 2,
  "heads": 2,
  "d_mlp": 256
}
```

- **Config rules:** `d_model` must be divisible by `heads`, and `layers` must
  be at least 1. The parameter count follows from the shape.
- **Verifier code:** CI never runs `verifier.py`; a maintainer reads it before
  evaluating it. Keep it under 256 KiB and use only the repo's environment
  (Python, numpy, torch and this package).

## Protocol

The verifier is run as `python verifier.py <instances_dir> <answers_dir>`.
- **Input:** each `<instances_dir>/<id>/` holds `config.json` and float32
  `weights.npz`. `verification.model.load_instance` reads both.
- **Output:** write `<answers_dir>/<id>.json`, either
  `{"answer": "PROVED"}` or `{"answer": "COUNTEREXAMPLE", "sequence": [0, 1, 1, 0]}`.
- **Grading:**
  - PROVED is right only on a sound network.
  - A counterexample is right only if it is a real violating input.
  - A missing answer is unsolved; a false one is wrong. Either fails the run.
- **Time limit:** when time runs out the process is killed, and answers
  already written still count. Write to a temporary name, then rename.

## Practise

```bash
uv run mnnc evaluate verify-transformer-len4 models/verify-transformer-len4/<name>
```

Each evaluation uses a fresh seed, recorded in the result. Use `--seed` to
repeat one. `models/verify-transformer-len4/example-enumerate/` is a starting
point.
