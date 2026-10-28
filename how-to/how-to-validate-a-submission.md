# How to validate a submission

Point the command at the folder containing `config.json` and `model.pt`:

```bash
uv run mnnc validate-submission models/mnist/my-model
```

The parent folder identifies the benchmark in `models/<benchmark>/<name>` and
must agree with the `benchmark` in `config.json`. A folder directly under a known
benchmark folder (for example `/tmp/mnist/my-model`) follows the same rule.
For another location such as `./my-model`, the benchmark comes from the config.
The filenames must be exactly `config.json` and `model.pt`.

Validation checks the config and checkpoint, runs the benchmark in a child
process with a ten-minute timeout, and logs a JSON report including the score,
parameter count, and benchmark pass/fail. Exit codes are 0 for a passing result,
1 for a completed benchmark below its pass threshold, and 2 for invalid input
or an evaluation/update error. Checkpoints are loaded with `weights_only=True`;
submission Python code is never executed.

The default command does not change the leaderboard, HTML, or replays and needs
no GitHub authentication or particular branch. Image datasets may be downloaded
and cached in `data/`; use `--data-root <path>` to choose another cache location.

Maintainers can additionally update the local site:

```bash
uv run mnnc validate-submission models/mnist/my-model --update
```

This updates `docs/leaderboard.json`, game replays where applicable, and generated
HTML. Other leaderboard entries are preserved; rerunning replaces the same
`(benchmark, folder name)` entry, even when the config's display name differs.
Completed benchmark results, including below-threshold scores, can be recorded;
invalid inputs and scoring errors do not update the site. Artifacts are prepared
in a temporary directory before installation, with rollback on install failure.

`--update` does not commit, push, merge PRs, or promote branches. After reviewing
local changes on `submission`, commit and push the generated artifacts, promote
`submission` to `main` through the usual PR flow, and fast-forward `submission`
to `main`. Pages deploys on pushes to `main`.
