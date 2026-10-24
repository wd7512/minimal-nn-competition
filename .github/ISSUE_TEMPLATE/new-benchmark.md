---
name: New benchmark request
about: Propose a new problem for the smallest-network leaderboard
title: "[Benchmark] "
labels: benchmark request
assignees: ""
---

## The challenge in one sentence

<!-- e.g. "Given a chess position, predict the engine's score." -->

## What goes in and what comes out

<!-- Describe the question and the expected answer in plain language. -->

## Why it fits this suite

This suite asks for the smallest plain feed-forward network, not the most
accurate model. Explain why a `linear` or `mlp` model can make a real attempt.

## Fixed material

- Where the examples, positions, games, or seeds come from:
- Can they be shared publicly with every contestant? Yes / No
- Are contestants trained on exactly what they are scored on? (This suite has no held-out split.)
- Is the material stable and versioned? How?

## Scoring

- The single score, higher is better:
- Formula or rule, precisely:
- What a perfect score looks like:
- What a random guess scores:
- What a model that always gives the same answer scores:
- The fixed pass bar:
- A pass requires: all examples / all seeds / a threshold / other:

## Size tie-break

- What counts as "smaller": parameter count, file size, other:
- Is there room to solve this at several sizes?
- Does the pass bar leave something to squeeze?

## Reproducibility

- Fixed seeds or sources of randomness:
- How a result is kept identical across machines and runs:
- Anything that could make two runs disagree:

## Cost

- Approximate evaluation time:
- Approximate data size:
- Anything expensive that could be cut without changing the benchmark:

## Example and limits

- One working example entry (or a plan to produce one):
- The score that example achieves:
- What a high score does **not** prove:
- How could a clever contestant game this?

## Proposer checklist

- [ ] The task can be stated in one sentence.
- [ ] A plain `linear` or `mlp` model can attempt it.
- [ ] Everyone is scored on the same fixed material.
- [ ] The material can be shared publicly.
- [ ] There is one number, and higher is better.
- [ ] There is a fixed pass bar, not "top N".
- [ ] A constant or trivial model does **not** pass.
- [ ] Smaller models win ties, and the bar leaves room for that race.
- [ ] Results are reproducible with stated seeds.
- [ ] Evaluation takes minutes, not hours.
- [ ] I have a working example or a concrete plan for one.
- [ ] I have read `CONTRIBUTING.md` and the submission rules.

## Maintainer checklist

- [ ] Scope fits the suite (fixed problem, minimal feed-forward models).
- [ ] Bar checked: a trivial model fails; the example passes.
- [ ] Runtime and data size are acceptable for routine evaluation.
- [ ] Data can be redistributed.
- [ ] Decision: accept / needs more work / decline.
