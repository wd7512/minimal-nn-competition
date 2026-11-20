# Verification: the largest verified transformer

How large can a transformer be while a verifier still decides, within 5
minutes, whether a chosen token can ever be its next token?

## Benchmark

- **Submission:** a verifier that claims a model shape.
- **Networks:** evaluation builds 100 fresh networks of that shape: 50 sound,
  50 with 1 to 3 planted inputs where token 1 wins.
- **Run:** the verifier answers each network with PROVED or COUNTEREXAMPLE.
- **Grading:** answers are graded against the planted labels, and all 100
  must be right. Larger models rank first.
- **Tiers:** inputs are binary sequences of exactly 0 or 4 tokens after BOS.
  A length-16 tier (65,536 inputs) is work in progress.

## Construction

Construction uses numpy only, with no training. It is deterministic from
(shape, seed).
1. Draw a random GPT-2-scaled transformer.
2. Project each input's final residual onto a random direction.
3. Put a threshold below the top 1 to 3 projections (unsafe), or just above
   the top one (a sound near miss at the same distance).
4. Plant `logit_1 - logit_0 = alpha * (projection - threshold)` exactly,
   through the unembedding and the last layer's output bias.
5. Check that every input is at least 0.5 from a tie after float32 rounding.

## Example results (len4, 4-CPU container)

`example-enumerate` checks every input with one batched forward pass:

| Parameters | Solved | Verifier time |
| --- | --- | --- |
| 790,272 | 100 | 1.3 s |
| 6,304,256 | 100 | 8.7 s |
| 25,191,424 | 100 | 39.2 s |
| 100,714,496 | 88 | 300 s (limit) |

With 16 inputs, forward-pass speed sets the limit at about 25M parameters.

## Next steps

Length 16 has about 4,000 times more inputs. That is where proofs over whole
sets of inputs should beat enumeration. Plain interval bounds were sound but
far too loose (a true gap of 4.8 to 9.5 was bounded as -35 to +84), so
relational bounds such as zonotopes or CROWN, which alpha-beta-CROWN uses,
are the natural candidates.
