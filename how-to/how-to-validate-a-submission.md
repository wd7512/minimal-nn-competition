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
uv run mnnc download-mnist --output data
uv run mnnc download-fashion-mnist --output data
uv run mnnc evaluate <benchmark> models/<benchmark>/<name>
```

Image benchmarks use `data/mnist.pt` and `data/fashion-mnist.pt` by default;
each cache contains all 70,000 examples. The evaluator rejects caches that do
not contain exactly 70,000 samples. Legacy `*-train.pt` / `*-test.pt` caches are
not used. Snake Simple 8x8 needs no data.

Expected passes are `1.0` for MNIST and Fashion-MNIST, and `64.0` for Snake Simple 8x8.
Confirm the score and parameter count match the PR description. A perfect
image score is expected; verify the cache is the correct benchmark's data.
There is no held-out split: an entry must fit all 70,000 examples.

If an entry fails, revert it on `submission` and do not promote it.

## 5. Rebuild the leaderboard

```bash
uv run mnnc collect --replay-dir replays
uv run mnnc render leaderboard.json docs --replays replays
```

`collect` re-evaluates every model and is the authoritative rebuild. It also
records a seed-0 replay for each Snake Simple 8x8 submission. `replays/` is temporary;
commit only `leaderboard.json`, `docs/leaderboard.json`, and `docs/`.

## 6. Check the site

```bash
rm -rf /tmp/mnnc-site-check
uv run mnnc render leaderboard.json /tmp/mnnc-site-check --replays docs/games
uv run mnnc render-speed speed-leaderboard.json /tmp/mnnc-site-check/internal
uv run mnnc render-internal internal-leaderboard.json /tmp/mnnc-site-check
diff -ru docs /tmp/mnnc-site-check
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
