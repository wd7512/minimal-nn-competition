# minimal-nn-competition

Find the smallest possible neural network for a problem.

**[View the leaderboard →](https://wd7512.github.io/minimal-nn-competition/)**

Problems: **Snake Simple 8x8**, **Snake Simple 16x16**, and **Snake Obstacles 8x8** (fill every free cell) · **MNIST** and **Fashion-MNIST** (classify images) · **Chess evaluation 10k** and **Chess evaluation 100k** (predict Stockfish within 0.1 pawns) · **Control tasks** (CartPole, Pendulum, and LunarLander).
Rules: feed-forward `linear` or `mlp` models, a standard activation, and a PyTorch `state_dict`.

Snake Obstacles 8x8 places two or four deterministic blocked cells per seed. A
run passes only after filling every free cell and then making one extra loop move
into the vacating tail. Layouts are checked by a closed-tour validator.

MNIST and Fashion-MNIST each use one 70,000-example dataset. Chess evaluation
uses fixed 10,000- and 100,000-position Stockfish sets; the 100k set includes
every 10k position. Control tasks use fixed seeds and a hand-rolled engine; no
data download is needed. There is no held-out split: train on the same examples
you are scored on and fit every example exactly.

## Submit

Open a pull request against `submission` with exactly
`models/<benchmark>/<name>/config.json` and `model.pt`. Do not change code,
dependencies, workflows, the leaderboard, or `docs/`.

- [How to make a Snake submission](how-to/how-to-make-a-snake-submission.md)
- [How to make an MNIST submission](how-to/how-to-make-an-mnist-submission.md)
- [How to make a Fashion-MNIST submission](how-to/how-to-make-a-fashion-mnist-submission.md)
- [How to make a Chess evaluation submission](how-to/how-to-make-a-chess-eval-submission.md)
- [How to make a Control (RL) submission](how-to/how-to-make-an-rl-submission.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and
[how to validate a submission](how-to/how-to-validate-a-submission.md).
