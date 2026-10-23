# How to validate a submission

Use this checklist for every pull request targeting `submission`. The
`validate` workflow checks paths and `config.json` only; it does not run the
model.

## 1. Check the pull request

```bash
gh pr view <number> --json files,mergeable,mergeStateStatus
gh pr checks <number>
```

Confirm:

- the PR adds one `models/<benchmark>/<name>/` directory;
- it contains only `config.json` and `model.pt`;
- `validate` passes;
- `config.json` has a non-empty `name` and `author`, and its benchmark matches
  the directory.

Do not execute code from the PR. The evaluator only loads `model.pt` with
`weights_only=True`.

## 2. Update `submission`

```bash
git fetch origin
git switch submission
git merge --ff-only origin/main
git push origin submission
```

Starting from the latest `main` avoids stale leaderboard and `docs/` conflicts.

## 3. Merge the submission

```bash
gh pr merge <number> --merge --delete-branch
git pull --ff-only origin submission
```

For several PRs, merge them all and evaluate once.

## 4. Evaluate

```bash
uv sync --dev --locked
uv run mnc download-mnist --output data
uv run mnc download-fashion-mnist --output data
uv run mnc evaluate <benchmark> models/<benchmark>/<name>
```

Image benchmarks use `data/mnist-test.pt` and
`data/fashion-mnist-test.pt` by default. Snake needs no data.

Expected passes are `1.0` for MNIST and Fashion-MNIST, and `64.0` for Snake.
Confirm the score and parameter count match the PR description. A perfect
Fashion-MNIST score is suspicious: verify it is not MNIST data.

If an entry fails, revert it on `submission` and do not promote it.

## 5. Rebuild the leaderboard

```bash
uv run mnc collect --replay-dir replays
uv run mnc render leaderboard.json docs --replays replays
```

`collect` re-evaluates every model and is the authoritative rebuild. It also
records a seed-0 replay for each Snake submission. `replays/` is temporary;
commit only `leaderboard.json`, `docs/leaderboard.json`, and `docs/`.

## 6. Check the site

```bash
rm -rf /tmp/mnc-site-check
uv run mnc render leaderboard.json /tmp/mnc-site-check --replays docs/games
uv run mnc render-speed speed-leaderboard.json /tmp/mnc-site-check/internal
uv run mnc render-internal internal-leaderboard.json /tmp/mnc-site-check
diff -ru docs /tmp/mnc-site-check
```

Commit after the diff passes:

```bash
git add leaderboard.json docs
git commit -m "Add <name> to leaderboard"
git push origin submission
```

## 7. Promote to `main`

```bash
gh pr create --base main --head submission \
  --title "Promote submission leaderboard updates"
```

Wait for CI, then merge:

```bash
gh pr merge <number> --merge
git fetch origin
git merge --ff-only origin/main
git push origin submission
```

Confirm the Pages deployment succeeds and the new rows appear at
<https://wd7512.github.io/minimal-nn-competition/>.
