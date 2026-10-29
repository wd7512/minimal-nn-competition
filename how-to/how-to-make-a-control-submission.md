# Make a control submission

The current controls use actual Gymnasium physics and native recorded video.
They are competition variants with wider initial states, not stock Gymnasium
scores. [Legacy results](../docs/legacy-controls/index.html) remain separate;
old checkpoints cannot be submitted unchanged.

## Contract

Evaluation uses exactly seeds **0–99**, CPU float32 inference and the pinned
Gymnasium 1.2.3 / Box2D 2.3.10 / NumPy 2.5.4 / pygame 2.6.1 stack on Python 3.12.
The [versioned manifest](../src/minimal_nn_competition/benchmarks/common/control/spec.json)
is authoritative. Results include its evaluation fingerprint, seed returns,
steps, termination/truncation flags and trajectory hashes. Train on seeds ≥100;
the example trainer does this automatically. Use 10000–10099 for sanity checks.

| Benchmark ID | Inputs → outputs | Wider initial conditions | Horizon | Score / pass |
|---|---|---|---|---|
| `cartpole-gym-v1` | 4 → 2 action logits | Uniform [-0.10, 0.10] for all four reset coordinates | 500 | Fraction surviving 500 steps without termination; all 100 must pass |
| `pendulum-gym-v1` | 3 → **1 continuous torque** | Angle uniform [-π, π], angular speed uniform [-2, 2] | 200 | Fraction with native return ≥ -500; all 100 must pass |
| `lunarlander-gym-v1` | 8 → 4 action logits | Native terrain/reset, plus independent uniform [-500, 500] extra force per axis before the native reset step | 1000 | Safe landing fraction; all 100 must land within the limit |

CartPole reset angles remain below the native ±12 degrees termination boundary. Native
position, reward and termination rules remain intact. Pendulum keeps native
observations `(cos θ, sin θ, θ̇)` and a continuous `Box([-2], [2])` action. Lander
keeps native Box2D terrain, contacts, leg joints, normalized observations,
engines, rewards and termination. Its initial force is the sum of the native
U(-1000,1000) and extra U(-500,500), a convolution supported on [-1500,1500].
The small subclass relies on the pinned native reset calling `step(0)` before
returning its observation and initializing shaping; it never overwrites a
returned state or changes subsequent physics.

Discrete outputs use first argmax. Lander action IDs are **0 idle, 1 left engine,
2 main engine, 3 right engine**. Pendulum emits a float32 tensor `(batch, 1)`;
for each observation, supply one finite torque, clipped directly to [-2, 2].
NaN, infinity and wrong output shapes are rejected. No discrete torque wrapper,
extra scaling or submitted Python code is allowed.

Every episode stops on native `terminated or truncated`; the final reward is
included. CartPole termination on step 500 fails. Pendulum truncates at 200.
Lander success requires native episode termination without a hull crash, a
settled (sleeping) lander, and both leg positions between the helipad posts
(inclusive), within 1000 steps / 20 simulated seconds. A timeout or crash fails,
even with a high return. It is not necessary for both contact flags to be true:
native settled/no-crash state determines stability. All three scores are the
fraction of successful seeds; every seed must succeed to pass overall. Mean
return is secondary. Legacy scores remain separate.

## Save the checkpoint

Submit only `models/<benchmark-id>/<name>/config.json` and `model.pt`, targeting
the `submission` branch. Example minimal Pendulum metadata:

```json
{
  "benchmark": "pendulum-gym-v1",
  "name": "my-policy",
  "author": "your-name",
  "architecture": "linear",
  "input_size": 3,
  "output_size": 1
}
```

Use the dimensions in the table for other tasks. Linear state dicts contain
`weight` and `bias`; the minimal Pendulum shapes are `(1, 3)` and `(1,)` (four
parameters). MLPs declare `hidden_sizes` and `activation` (`relu` or `identity`)
and use Sequential keys (`0.weight`, `0.bias`, `2.weight`, …). Store every tensor
as float32. Checkpoints are loaded on CPU with `weights_only=True` and strict
key, tensor, shape and size validation.

## Train, score and record native video

Install FFmpeg with libx264 (`sudo apt-get install ffmpeg` on Ubuntu), then:

```sh
uv sync --dev --locked
uv run mnnc train-cartpole
uv run mnnc train-pendulum --iterations 500 --device cpu
uv run mnnc train-lunarlander --iterations 500 --device cpu
uv run mnnc evaluate pendulum-gym-v1 models/pendulum-gym-v1/example
uv run mnnc validate-submission models/pendulum-gym-v1/example --update
```

CartPole writes a small fixed linear policy; the other examples train native PPO
with shared code. Pendulum uses a Gaussian torque policy during training and its
mean during deterministic evaluation; exploration parameters are not part of the
submission. Training randomness is separate from the fixed evaluation starts.
Reference results are measured after training; success is never assumed.

To evaluate every current model and record one seed-0 episode for each control:

```sh
uv run mnnc collect --models models --output docs/leaderboard.json --replay-dir docs/games
uv run mnnc render docs/leaderboard.json docs --replays docs/games
```

Other benchmark families need their documented datasets. Control recording uses
the same rollout as scoring, streams actual `rgb_array` frames to H.264 MP4,
native 50/30/50 fps, CRF 28, yuv420p and faststart, with no audio. The static site
uses normal browser video controls; there is no browser physics simulator.
Only seed 0 is recorded by default; all 100 seeds still count for scoring.
On headless Linux, set `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` if necessary.

See [local validation and PR instructions](how-to-validate-a-submission.md).
