# Legacy control benchmarks

These checkpoints and the [archived control leaderboard](../docs/legacy-controls/index.html)
use the bespoke pre-Gymnasium simulators. They cannot be interpreted as native
Gymnasium v1 submissions. Scores remain read-only and are excluded from current
collection by living outside `models/`.

The retired physics, renderer and trainers are available at commit `e6342a2`.
Current IDs are `cartpole-gym-v1`, `pendulum-gym-v1`, `lunarlander-gym-v1`.
Native Lander action IDs differ, Pendulum now has one continuous torque output,
and all three tasks use different physics/reset contracts. Re-training and full
evaluation are required; changing a config ID is not a migration.
