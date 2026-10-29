# Four-benchmark breakability experiment — 2026-10-29

## Scope and reproducibility

Experimental branch: `experiment/breakability-proof-gpt6-20261029`; draft PR #235.
Workflow: [GitHub Actions run 40937766132](https://github.com/wd7512/minimal-nn-competition/actions/runs/40937766132).
Script: `experiments/breakability_proof.py`.
Environment: GitHub Actions Ubuntu, `uv sync --dev --frozen`, repository's pinned Gymnasium/Box2D stack.
Evaluation: **100 fixed seeds 0–99**, exact repository benchmark evaluator. The best candidate per task also went through `python -m minimal_nn_competition validate-submission` without `--update`.
All trained trajectories used seeds >=100. Public leaderboard and submitted models were left untouched.
Checkpoints, config files and JSON reports are attached to the workflow as four `proof-*` artifacts.

## Results

| Benchmark | Incumbent | Best experiment | Parameters | Official validation | Leader beaten? |
|---|---|---|---:|---|---|
| CartPole Gym v1 | 100/100, 10 params | 100/100 | **9** | Exit 0 | **YES** |
| Pendulum Gym v1 | 100/100, 4,481 params | 100/100 | **1,217** | Exit 0 | **YES** |
| Snake obstacles 8×8 | 19/100, 145,156 params | 19/100 | 145,156 | Exit 1 (below 100% pass requirement) | **NO** |
| LunarLander Gym v1 | 95/100, 4,996 params | 94/100 | 1,476 | Exit 1 (below 100% pass requirement) | **NO** |

**The validator's exit 1 on Snake and LunarLander means below the benchmark's global all-100-seeds pass requirement, not a failure to load or score the model.** The baseline LunarLander at 95/100 would also not globally pass.

### CartPole: exact decision-function reduction

Given the incumbent's linear output logits `z0(x)`, `z1(x)`, construct a `4 -> ReLU(1) -> 2` MLP whose hidden activation is `h = ReLU(z1-z0)` and outputs `(0, h)`. With argmax tie-breaking toward action 0, this has the same action decision as the incumbent in exact arithmetic, but **9 rather than 10 parameters**. Verified 100/100 seeds and validator exit 0. This is a reparameterisation, not gradient training.

### Pendulum: teacher imitation and policy distillation

Generated native Gymnasium rollouts using the incumbent teacher with seeds 100–179, plus random pendulum observation samples. Fitted small ReLU students by Adam to clipped teacher torques, and evaluated the original repository's reward threshold on seeds 0–99.

| Architecture | Parameters | Evaluation pass rate |
|---|---:|---:|
| 3–8–8–1 | 113 | 80/100 |
| 3–16–16–1 | 353 | 80/100 |
| **3–32–32–1** | **1,217** | **100/100** |

Best student: mean reward about -139.48; official validator exit 0. This is a **72.8% reduction in parameters**. These results apply to the **current reward >= -500** pass rule, *not* the proposed two-second upright stability rule.

### LunarLander: imitation learning

Collected native Box2D trajectories from incumbent teacher on seeds 100–299; trained 8–width–width–4 ReLU students with logit MSE and cross-entropy.

| Architecture | Parameters | Evaluation pass rate |
|---|---:|---:|
| 8–32–32–4 | 1,476 | **94/100** |
| 8–48–48–4 | 2,980 | 93/100 |
| 8–56–56–4 | 3,924 | 92/100 |

None beat the incumbent's 95/100, despite much smaller networks. Strongest next step: fine-tune the 32-width student with on-policy rollouts, use DAgger or PPO with safe-landing success as the objective, and evaluate on the fixed 100-seed contract only at the end.

### Snake obstacles: final-layer bias perturbation

Tested 48 random final-bias perturbations at four scales on seeds 100–139. All scored 0/40 on those training seeds. Re-evaluated the eight top perturbations on the official 100 seeds: seven scored 19/100, one scored 18/100. All retained 145,156 parameters. **No win.**

Because the leaderboard uses game length to break ties among failed Snake submissions, even equal success and equal parameter count do not demonstrate a rank improvement. Next step: change policy class or train an obstacle-aware planning teacher rather than tweaking output biases. A substantially smaller network matching 19/100 could also be interesting, but game-length tie-breaking must be checked.

## Conclusion

**2 of 4 leaders demonstrably beaten with actual repository evaluation and validator runs; 2 not yet beaten.** The earlier qualitative ranking was directionally correct for CartPole and Pendulum but overconfident for Snake obstacles and LunarLander. No benchmark results have been added to the public leaderboard. This experiment PR is deliberately a draft; its artifacts are for review, not automatic promotion.
