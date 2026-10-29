# Gymnasium control migration (#229)

29 October 2026. The investigation has been carried through to benchmark
replacement, fresh training, evaluation, recorded native episodes and a rebuilt
leaderboard. The original investigation-only scope was expanded by the owner.

## Frozen evaluation contract

The [manifest](../src/minimal_nn_competition/benchmarks/common/control/spec.json)
fixes Python 3.12, Gymnasium 1.2.3, Box2D 2.3.10, pygame 2.6.1, NumPy 2.5.4,
seeds **0 through 99**, native episode limits and the following competition starts:

| ID | Network | Reset distribution | Limit | Score / pass |
|---|---|---|---|---|
| `cartpole-gym-v1` | 4 inputs, 2 logits | U(-0.10, 0.10) for each coordinate | 500 | Survival fraction; all 100 must survive without native termination |
| `pendulum-gym-v1` | 3 inputs, **1 continuous torque** | Angle U(-pi, pi), angular speed U(-2, 2) | 200 | Fraction with native return >= -500; all 100 must pass |
| `lunarlander-gym-v1` | 8 inputs, 4 logits | Native reset plus independent U(-500, 500) extra force on each axis before the reset physics step | 1000 | Safe landing fraction; all 100 must land within 1000 steps / 20 seconds |

Reset bounds and Pendulum threshold were fixed before training the references.
The landing-based Lander criterion was agreed before merging the migration. There are
no difficulty strata. CartPole starts remain inside the native +/-12-degree
termination angle. Pendulum takes one finite float32 output and clips directly
to [-2, 2]. Native Lander IDs are 0 idle, **1 left, 2 main**, 3 right. Native
observations, physics, rewards, contacts and termination are preserved. Stop on
`terminated or truncated`, including the final reward. CartPole termination on
step 500 fails. Lander succeeds only on native termination without a hull crash,
with its body settled/asleep and both leg positions between the helipad posts
(inclusive) within 1000 steps / 20 simulated seconds. Timeouts, hover and crashes
fail regardless of return. The score is the landing fraction; all 100 must
succeed to pass overall. Mean native return is secondary.

The Lander subclass adds force only while native `prev_shaping is None`, before
reset's `step(0)`. Native reset then computes the actual observation and shaping
with all leg joints already assembled. The combined initial force is a
convolution with support [-1500, 1500], not a uniform over that interval. Tests
check state-derived observations, shaping, two joints and subsequent identical
native engine transitions. This initialization hook depends on pinned private
reset sequencing; revalidate it before upgrading Gymnasium. Prefer a supported
upstream force-range option if one becomes available.

Every result stores the complete contract and a SHA-256 fingerprint, per-seed
returns/steps, termination/truncation flags and hashes of complete trajectories.
Custom seed/horizon evaluations receive a different fingerprint. Runtime checks
reject dependencies inconsistent with the frozen spec. These are competition
variants, not untouched stock Gymnasium benchmark scores.

## Fresh reference results

| Reference | Parameters | Official score | Mean return | Pass | 100-seed evaluation seconds | Seed-0 MP4 bytes |
|---|---:|---:|---:|---|---:|---:|
| CartPole | 10 | 100% | 500.00 | yes | 0.472 | 27,424 |
| Pendulum | 4,481 | 100% | -138.01 | yes | 0.460 | 19,340 |
| LunarLander | 4,996 | 95% (95/100 landings) | 279.95 | no | 0.813 | 95,030 |

CartPole uses a regenerated fixed linear reference, evaluated against the new
native contract. Pendulum and Lander were freshly trained for 500 PPO iterations,
8 episodes per iteration, CPU with one torch thread, seed 0. Reset seeds were
100-4099, disjoint from evaluation. Pendulum uses a fixed-unit-variance Gaussian
for training and its mean for deterministic inference. All policies are standard
validated float32 state dicts; no submitted executable code is accepted. A
minimal 4-parameter `nn.Linear(3,1)` Pendulum checkpoint is also tested.

Held-out sanity seeds **10000-10099** (not used to optimize the references):
CartPole 100% / 500.00 mean return; Pendulum 100% / -128.10; Lander 92% successful landings / 270.68 mean
return. CartPole and Pendulum pass; Lander falls short of the all-seeds requirement. The official leaderboard
continues to use only 0-99. Timings above exclude installation/training/video and
are observations on this Linux x86-64 worker, not CI performance promises.

Two separate production `mnnc collect` processes produced byte-identical
300-episode result ledgers. Full rendered seed-0 rollouts matched scoring hashes
and rewards for each reference, including engine activity. Independent native
transition tests compare observations, rewards and termination; time-limit tests
verify Pendulum ends at exactly 200 steps. Exact equality across operating
systems or different Box2D builds is not established: use the pinned Linux stack
for canonical scoring.

## Native playback and publishing

The shared rollout streams actual Gymnasium RGB frames to FFmpeg. MP4 uses H.264,
CRF 28, yuv420p, faststart and no audio, at native 50/30/50 fps. The seed-0 Lander example reaches the time limit
and is correctly a failed episode, despite its positive return. All three full
videos total **141,794 bytes**, below the 2 MiB-per-episode budget. FFmpeg decoded
every file; ffprobe verified codec, pixel format, native fps and exactly
`seed_steps + 1` frames, including initial and final states. Chromium 154, served
over local HTTP, loaded, played and sought within all three videos. Seed 0 is the only
default demonstration; all 100 seeds still count for scoring.

Collection writes MP4 directly. Static submission pages use standard browser
video controls, without control-specific JavaScript, approximate drawing or
browser physics. In-place rendering and publishing to another directory preserve
the same video bytes. The active leaderboard and all control detail pages were
regenerated from the new results. Pages publishes these committed artifacts when
the PR is promoted through dev to main.

## Compatibility and source reduction

The Gymnasium benchmarks replace the previous controls completely. Old simulators,
checkpoints, results, replay pages and investigation artifacts are removed.
Only the current Gymnasium models, results and native videos are published.

Removed/replaced maintained source:

- Three bespoke simulator implementations: now tiny versioned registrations
  backed by one shared Gymnasium adapter.
- `common/control/replay.py`: removed, replaced by one streamed native MP4 encoder.
- All control canvas drawing, replay JavaScript and control replay HTML machinery
  from `leaderboard/render.py`: removed; snake replays remain.
- Three duplicated example trainers: replaced by one shared native trainer.
- Bespoke physics/replay tests: replaced by native transition, reset, scoring,
  continuous-action validation, deterministic rollout, video and publishing tests.
- The investigation-only pilot script and test module: removed in favour of the
  production path and normal CI coverage.

Source measurements against pre-investigation `0e3e3fc`:

| Scope | Before | After | Change |
|---|---:|---:|---:|
| `uv run dev-tools tokens` (package rules-v2 tokens) | 21,471 | 19,640 | **-1,831 (-8.5%)** |
| All tracked first-party Python/JS/TS files | 89 | 87 | -2 |
| Nonblank, non-comment source LOC | 11,844 | 10,749 | **-1,095** |
| Python rules-v2 tokens including scripts/tests | 48,762 | 45,950 | **-2,812** |

Count all tracked `.py`, `.js`, `.ts`, including scripts/tests. Exclude generated
`docs/`, agent configuration `.opencode/`, checkpoints/media, lockfiles, external
and vendored dependencies, ignored environments and local scratch. There are no
standalone JS/TS source files here. Embedded JS is counted once inside Python;
source LOC includes string bodies, making removal of embedded drawing code
visible even though Python token scoring treats a string as one literal.

Recount from each checkout (stage new source first):

```sh
uv run dev-tools tokens
uv run python - <<'PY'
import logging
import subprocess
from pathlib import Path
from minimal_nn_competition.internal.scoring import score_files
files = [Path(p) for p in subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
         if Path(p).suffix in {'.py', '.js', '.ts'}
         and not p.startswith(('docs/', '.opencode/'))]
loc = sum(sum(bool(s.strip()) and not s.lstrip().startswith(('#', '//'))
              for s in p.read_text().splitlines()) for p in files)
logging.basicConfig(level=logging.INFO)
logging.info('files=%d source_LOC=%d Python_tokens=%d', len(files), loc,
             score_files([p for p in files if p.suffix == '.py']))
PY
```

The irreducible glue is reset configuration, finite action/shape validation,
strict checkpoint inference, score aggregation/fingerprinting and video encoding.
There is no replacement simulation framework. Required runtime dependencies are
pinned; FFmpeg is needed for recording, not for rendering existing site assets.

## Reproduce

See the [submission guide](../how-to/how-to-make-a-control-submission.md) for the
complete contract and commands. Install FFmpeg/libx264, then `uv sync --dev
--locked`. Use `mnnc train-cartpole`, `train-pendulum --iterations 500 --device cpu`
and `train-lunarlander --iterations 500 --device cpu`. All model configs/checkpoints
and measured official results are committed. Native tests run in ordinary CI;
there is no optional Gymnasium skip. Isolated scoring drains its result pipe
before joining the worker, so the larger trajectory metadata cannot deadlock
on a full IPC buffer; a native-result regression test covers this path. Run `uv run dev-tools check` and the normal
CI leaderboard regeneration/diff before promotion.
