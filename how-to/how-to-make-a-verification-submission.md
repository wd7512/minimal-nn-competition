# How to make a verification submission

Verification turns the competition around: you submit a *verifier*, and you
win by verifying the *largest* transformer. Your config claims a model shape.
Evaluation builds 100 fresh random transformers of that shape and runs your
verifier once over all of them. Of the 100, 50 are sound: token 1 is never the
next token for any input. The other 50 hide 1 to 3 inputs where token 1 is
the next token. The submission passes only when all 100 are answered correctly
within 5 minutes in total. Larger parameter counts rank first.

| Benchmark | Inputs per network | Status |
| --- | --- | --- |
| `verify-transformer-len0` | 1 (just the BOS token) | Trivial warm-up |
| `verify-transformer-len4` | 16 (every 4-token binary sequence) | Main tier |
| Length 16 | 65,536 | Work in progress |

## The networks

- Vocabulary `{0, 1}` plus a BOS token (id 2) at position 0. Every input is
  BOS followed by exactly `max_len` tokens.
- Decoder-only transformer with no LayerNorm:
  - learned token and position embeddings;
  - `layers` blocks, each with causal multi-head attention (`wq`, `wk`, `wv`
    and `wo`, no biases, scale `1/sqrt(d_model/heads)`) and a ReLU MLP
    (`w1`, `b1`, `w2` and `b2`), each added back to the residual stream;
  - an `unembed` matrix with no bias.
- The predicted token is the argmax of the last position's logits.
- `minimal_nn_competition/verification/forward.py` is the reference forward
  pass, and `model.py` lists every weight name and shape.
- Each network is built so that every input's logit gap is at least `tau`
  from a tie. `tau` is given in the instance, so float64 rounding cannot
  change any answer.

## 1. Create the submission

Open a pull request against the `submission` branch with exactly two files:

```text
models/verify-transformer-len4/<name>/config.json
models/verify-transformer-len4/<name>/verifier.py
```

`verifier.py` is code. CI never runs it; a maintainer reads it before
evaluating it locally. Keep it under 256 KiB and use only the repo's
environment (Python, numpy, torch and this package).

## 2. Write `config.json`

```json
{
  "benchmark": "verify-transformer-len4",
  "name": "example",
  "author": "example author",
  "architecture": "verifier",
  "d_model": 64,
  "layers": 2,
  "heads": 2,
  "d_mlp": 256,
  "description": "Optional."
}
```

The parameter count follows from the shape and the benchmark. `d_model` must
be divisible by `heads`, and `layers` must be at least 1.

## 3. Write `verifier.py`

It is run once as `python verifier.py <instances_dir> <answers_dir>`.

- **Input:** each `<instances_dir>/<id>/` holds `config.json` (shape,
  `forbidden`, `tau`, `id`) and `weights.npz` (float32).
  `minimal_nn_competition.verification.model.load_instance` reads both.
- **Output:** write `<answers_dir>/<id>.json` for each network.
- **Write safely:** write to a temporary name and then rename it. When time
  runs out the process is killed; answers already written still count.

Answers use one of three forms:

```json
{"id": "007", "answer": "PROVED", "leaves": [
  {"box": [[0, 1], [0, 1], [0], [0, 1]], "witness": 0, "method": "interval"},
  {"box": [[1], [1], [1], [1]], "witness": 0, "method": "point"}]}
{"id": "042", "answer": "COUNTEREXAMPLE", "sequence": [0, 1, 1, 0]}
{"id": "013", "answer": "UNKNOWN"}
```

- **Box:** a box lists the allowed tokens at each input position. A PROVED
  answer's leaves must cover every input exactly once.
- **Interval leaf:** proved with rigorous interval bounds, which the checker
  recomputes. It shows that `witness` beats token 1 on every sequence in the
  box.
- **Point leaf:** a single sequence, where the float64 forward pass must give
  `logit_witness - logit_1 >= tau / 2`.
- **Counterexample:** the float64 forward pass must put token 1 ahead of every
  other token by at least `tau / 2`.
- **Outcomes:** a rejected claim is wrong and fails the submission. UNKNOWN or
  a missing answer is unsolved, which also fails it.

## 4. Practise locally

```bash
uv run mnnc make-instances verify-transformer-len4 practice --d-model 64 --layers 2 --heads 2 --d-mlp 256
```

```bash
uv run mnnc check-answer practice/instances/000 answer.json
```

`practice/labels.json` holds the ground truth for practice only. To run the
full evaluation:

```bash
uv run mnnc evaluate verify-transformer-len4 models/verify-transformer-len4/<name>
```

Every evaluation uses a fresh random seed, which is recorded in the result;
pass `--seed` to reproduce one. The examples in
`models/verify-transformer-len0/example-interval/` and
`models/verify-transformer-len4/example-enumerate/` are starting points.
