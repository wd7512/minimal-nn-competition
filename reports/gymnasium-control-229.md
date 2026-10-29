# Gymnasium controls: investigation #229

29 October 2026. **Design candidate for review, not a benchmark migration.**
Native Box2D/classic-control episodes work, repeat exactly on this Linux x86-64
worker, and render through one shared RGB-to-MP4 path. No production benchmark,
checkpoint, trainer, replay or leaderboard is changed. Do not compare these scores
with legacy scores or advertise stock Gymnasium comparability.

[Native video samples and full measurements](../docs/investigations/gymnasium-229/index.html)
are committed static assets; Pages serves this path after promotion through dev
to main. The page uses browser video controls, without JavaScript or custom physics.

## Candidate evaluation contract

Use IDs `cartpole-gym-v1`, `pendulum-gym-v1`, `lunarlander-gym-v1` in the follow-up.
Exactly seeds **0 through 99 inclusive**, one uniform reset policy per task, no
strata. Training/dev seeds start at 100; sanity checks use 10000–10099. Freeze the
contract before training or accepting submissions. These bounds were tested with
a zero-output policy; they are candidates, not claims of calibrated difficulty.

| Task | Native interface and initialization | Limit | Proposed score and passed |
|---|---|---|---|
| CartPole-v1 | 4 observations → 2 logits, first argmax; `reset(options={"low": -0.1, "high": 0.1})` for every coordinate | 500 | Fraction reaching 500 steps **without termination**; passed iff all 100 succeed |
| Pendulum-v1 | 3 observations → 1 torque; `reset(options={"x_init": π, "y_init": 2.0})` | 200 | Fraction with native episode return ≥ -500; passed iff all 100 succeed; threshold needs reference-policy calibration before adoption |
| LunarLander-v3 | 8 observations → 4 logits, first argmax; native initialization plus extra force described below | 1000 | Mean native return; passed iff mean ≥ 200 over all 100 episodes, not a per-episode landing claim |

Always stop on `terminated or truncated`, include the final transition's reward,
and retain every seed's return, steps and both flags. A CartPole failure on step
500 does not pass. Publish mean return as a secondary metric for CartPole and
Pendulum. LunarLander's score is deliberately a mean, not a binary landing fraction;
its versioned leaderboard must label the units.

Use default native kwargs and time limits; LunarLander is discrete,
`gravity=-10.0, enable_wind=False`. Native action IDs are 0 idle, **1 left engine,
2 main engine**, 3 right engine. Old checkpoints are incompatible.

Pendulum inference accepts a float32 batch output of shape `(1, 1)`, extracts
one unbatched `(1,)` array, rejects NaN/Inf and wrong shapes, then clips directly
to [-2, 2] (no argmax or extra scaling). The pilot tests this unbatched boundary.
Existing `nn.Linear(3, 1)` materialization already supports a minimal 4-parameter
model: state-dict `weight` shape `(1, 3)`, `bias` shape `(1,)`. MLPs retain the
existing Sequential key convention and a final width 1. Keep strict CPU,
weights-only checkpoint loading, tensor/key/size checks and arbitrary-code ban.
Replace Pendulum's categorical PPO hook with a continuous policy trainer in a
separate change; do not reinterpret a five-output checkpoint as continuous torque.

## LunarLander reset strategy

Gymnasium 1.2.3 ignores reset options. Its reset creates terrain, bodies and leg
joints, applies force U(-1000, 1000) on each axis, then calls `self.step(0)` to
advance Box2D and initialize shaping. `WideLander` adds independent U(-500, 500)
force on each axis **only when `prev_shaping is None`**, immediately before that
native step. The sum is a convolution of two uniforms with support [-1500, 1500],
not a uniform over that interval. No module constants, body transforms, returned
observations, renderer or step equations are replaced. It consumes two additional
native RNG draws; all later randomness follows that declared stream.

Tests verify two leg joints, state-derived reset velocity/angle, reset shaping,
and identical subsequent engine transitions when bypassing the subclass.
The hook depends on private reset sequencing: pin the version and revalidate on
upgrades. Prefer an upstream supported force-range option when available. Retaining
stock starts would be simpler but would not implement the decided wider range.
No stronger tilt/velocity mutation or copied reset implementation is needed.

## Reproduce and inspect

Run from the repository root with Python 3.12 and FFmpeg with libx264 installed:

```sh
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run scripts/pilot_gymnasium.py docs/investigations/gymnasium-229
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run scripts/pilot_gymnasium.py work/gymnasium-repeat --verify docs/investigations/gymnasium-229/results.json
uv run --no-project --python 3.12 --with 'gymnasium[box2d]==1.2.3' --with 'numpy==2.2.6' --with 'pytest==9.1.1' python -m pytest -o addopts='' tests/test_gymnasium_pilot.py -q
```

The script pins Gymnasium 1.2.3, NumPy 2.2.6, pygame 2.6.1, Box2D 2.3.10 and
SWIG 4.5.1 without adding package runtime dependencies. Tests are opt-in when
Gymnasium/Box2D are absent; run them with the pinned pilot environment above.
`results.json` contains the dependency/reset/seed/scoring/video manifest, its
SHA-256 fingerprint, host/Python version, all initial observations and complete
trajectory hashes (observations, rewards, termination and truncation flags).
The current candidate fingerprint is recorded there; changing it invalidates a
cross-process comparison. This identifies a pilot contract, not an approved
submission evaluation version.

All 600 unique episodes repeated exactly in the same process and a separate
process: 2400 scoring rollouts total, plus seed-0 capture in each process.
All 100 initial states were distinct for each profile. CartPole challenge angles
were [-0.0960, 0.0962] radians, safely within ±0.20944; Pendulum angular speeds
were [-1.9988, 1.9690]. Lander initial horizontal velocity observations widened
from [-0.7995, 0.7968] to [-1.0784, 1.0829]. Rendering did not change seed-0
trajectories. Exact equality across operating systems/Box2D builds is **not**
established; canonical scoring should use a fixed Linux runner/build.

| Zero-output linear baseline | Stock return min / mean / max | Wider return min / mean / max | Stock / wider seconds per 100 |
|---|---|---|---|
| CartPole | 8 / 9.400 / 11 | 7 / 9.460 / 12 | 0.008 / 0.008 |
| Pendulum | -1898.940 / -1180.290 / -377.291 | -1892.391 / -1172.481 / -500.441 | 0.280 / 0.279 |
| LunarLander | -335.368 / -136.700 / -0.981 | -357.472 / -131.573 / 64.883 | 0.165 / 0.164 |

These are first-run measurements, not CI budgets or trained baselines. CartPole
and lander episodes terminated; every Pendulum episode truncated at exactly 200.
Legacy zero-weight torch model evaluation took 0.012 / 0.121 / 0.007 seconds per
100 respectively. Interfaces, horizons and episode lengths differ, and the spike
omits torch inference: this is a rough cost comparison, not a speed claim. The
full first pilot including repeats and video took 3.0 seconds after installation.
New dependency installation/build costs are excluded; normal package CI gains no
Gymnasium dependency. Reference-policy calibration and full-horizon timings are
required before migration; existing Pendulum/lander models cannot provide those.

Video: native 50 / 30 / 50 fps respectively, H.264 MP4, CRF 28, yuv420p,
`faststart`, no audio, one seed per environment. Files are 3,124 / 46,311 / 7,701
bytes (57,136 total). Target ≤2 MiB per full demonstration, lower CRF only if
readability needs it. All frames include native reset and final states; no
frame skipping or manually drawn scene. Short idle episodes are explicitly
labelled failures. Use a trained deterministic model for eventual featured videos.
MP4 is broadly browser-compatible; FFmpeg decoding is checked here, interactive
playback across browsers still needs review. Do not create 100 videos by default.

## Source footprint and deletion plan

At base `0e3e3fc`, `uv run dev-tools tokens` reports **21,471** package source
tokens (rules v2); unchanged after this spike. CartPole / Pendulum / LunarLander
scopes are 603 / 843 / 1,015, including trainers. This tool excludes scripts and
tests, so it alone cannot establish net simplification. This PR adds investigation
source; it makes **no claim of a net source reduction** and removes no files.
The wider inventory and exact recount command follow below.

Replace/delete in the migration PR, after native evaluation and playback pass:

- `benchmarks/{cartpole,pendulum,lunarlander}.py`: 420 lines, 1,260 tokens
  combined. Replace local reset/dynamics/scoring loops with a thin shared native
  rollout plus per-task reset/action/score constants; retain only loading hooks.
- `benchmarks/common/control/replay.py`: delete 142 lines / 513 tokens. Reuse the
  spike's RGB encoder path for deterministic loaded-model rollouts.
- `leaderboard/render.py`: delete `_CONTROL_REPLAY_JS` and control canvas/replay
  wrappers; use a video element and asset URL. Keep the separate snake renderer.
  The embedded JS is already counted once as Python source, never again as JS.
- `tests/test_control_replay.py`: retire 206 lines / 942 tokens after native capture
  tests replace them. In `test_rl_environments.py`, remove bespoke integration,
  reward and old bounds tests; retain validation/scoring tests adapted to v1.
- Replace duplicated trainer wrappers in `examples/*_submission.py`; keep common
  PPO infrastructure where useful. Update registry, runner, scope inventory and
  how-to docs; remove categorical Pendulum assumptions.

Promote only when deleted package/renderer source exceeds the permanent native
adapter and encoding glue. Do not keep legacy simulators in the installed package
indefinitely: archive legacy results, videos, configs and the old source commit
under explicitly labelled legacy links. Freeze old leaderboard data read-only;
retrain/revalidate against new IDs, regenerate videos, then change featured links.
An archive may keep historical custom replay pages as static artifacts without
keeping their generators or physics in maintained source.

Suggested PR order: (1) approve bounds, scores and pinned evaluation manifest;
(2) native adapters, continuous Pendulum validation/training and contract tests;
(3) reference calibration, legacy archive, shared capture, renderer/physics
removals and a measured net-token reduction. Reject a migration that adds an
abstraction framework or fails the source-count gate. No migration approval is
implied by merging this investigation.

Wider first-party inventory (same counting rules before/after):

| Scope | Before | After | Delta |
|---|---:|---:|---:|
| Tracked Python/JS/TS files | 89 | 91 | +2 |
| Nonblank, non-comment source LOC | 11,844 | 12,076 | +232 |
| Python rules-v2 source tokens, including scripts/tests | 48,762 | 50,115 | +1,353 |
| Installed package source tokens (`dev-tools tokens`) | 21,471 | 21,471 | 0 |

Count all tracked first-party `.py`, `.js`, `.ts`, including scripts/tests;
exclude generated `docs/`, agent configuration `.opencode/`, media/checkpoints,
lockfiles, vendored/external dependencies and untracked environments. There are
no standalone JS/TS source files in this snapshot; embedded renderer JS remains
inside Python source. LOC includes docstrings/string bodies, excluding blank
lines and lines beginning with `#` or `//`. Recount from each checkout with:

```sh
uv run dev-tools tokens
uv run python - <<'PY'
import subprocess
from pathlib import Path
from minimal_nn_competition.internal.scoring import score_files
files = [Path(p) for p in subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
         if Path(p).suffix in {'.py', '.js', '.ts'}
         and not p.startswith(('docs/', '.opencode/'))]
loc = sum(sum(bool(s.strip()) and not s.lstrip().startswith(('#', '//'))
              for s in p.read_text().splitlines()) for p in files)
import logging
logging.basicConfig(level=logging.INFO)
logging.info('files=%d source_LOC=%d Python_tokens=%d', len(files), loc,
             score_files([p for p in files if p.suffix == '.py']))
PY
```

Baseline is the tracked tree of `0e3e3fc`; stage new source files when measuring
a working tree. The experiment and its tests cost 881 and 472 tokens respectively;
reuse the small reset/action/video pieces and retire experiment-only reporting
when migrating, rather than maintaining two evaluation paths forever.
