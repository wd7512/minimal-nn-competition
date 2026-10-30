# How to make a submission

Submit a model by opening a pull request against the `submission` branch. Start
with the general steps below, then follow the guide for your benchmark.

## 1. Choose a benchmark

Read the matching guide for its input and output sizes, evaluation rules, and
checkpoint format:

- [Snake](how-to-make-a-snake-submission.md)
- [Image compression](how-to-make-an-image-compression-submission.md)
- [Chess](how-to-make-a-chess-submission.md)
- [Control](how-to-make-a-control-submission.md)

Submissions must use a feed-forward `linear` or `mlp` model and a PyTorch
`state_dict` with all tensors stored as `float32`. Do not submit Python code.

## 2. Prepare the two required files

Create a directory using the benchmark ID and a short submission name:

```text
models/<benchmark>/<name>/config.json
models/<benchmark>/<name>/model.pt
```

The configuration must identify the benchmark and include non-empty `name` and
`author` values, the model architecture, and the correct input and output sizes.
Add `hidden_sizes` and other options when the selected architecture or benchmark
requires them. The benchmark guide has examples and exact requirements.

## 3. Validate locally

From the repository root, run:

```bash
uv run mnnc validate-submission models/<benchmark>/<name>
```

Fix any reported errors before opening the pull request. The submission branch
accepts exactly `config.json` and `model.pt` in one new model directory; do not
include code, dependencies, workflows, leaderboard files, or documentation in
the submission pull request.

## 4. Open the submission pull request

Fork the repository, add the upstream remote, and branch from the latest
`upstream/submission`:

```bash
git remote add upstream https://github.com/wd7512/minimal-nn-competition.git
git fetch upstream
git switch -c submit/<name> upstream/submission
```

Commit and push the two model files to your fork, then open a pull request
targeting `submission`. Keep `origin` set to your fork and `upstream` set to this
repository. If `upstream/submission` advances before your pull request is
merged, rebase your branch onto its latest version and push the updated branch.

The leaderboard records the date when `config.json` was first added in Git
history. Keep the configuration committed in the submission branch so the
leaderboard can record a valid date for your entry.
