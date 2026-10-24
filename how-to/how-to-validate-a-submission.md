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

Only the image benchmarks need a download. Each cache holds all 70,000
examples, and the downloader verifies pinned SHA-256 digests, failing on any
mismatch. Snake needs no data; Chess uses its committed Stockfish sets. There
is no held-out split: an entry must fit every example.

Expected scores: `1.0` MNIST and Fashion-MNIST, `64.0` Snake Simple 8x8,
`256.0` Snake Simple 16x16 (100 seeds, 4096-move cap), `1.0` Chess (every
position within 0.1 pawns of Stockfish). Confirm the score and parameter count
match the PR description, and that an image cache is the correct benchmark's
data.

Snake Obstacles 8x8 has no fixed score; it is the mean final length over
seeded obstacle layouts. Inspect `seed_passed` and `target_lengths` in the
result, and validate the layouts separately:

```bash
uv run mnnc validate-snake-obstacles-8x8 --seeds 100
```

The command exits non-zero on any failure.

If an entry fails, revert it on `submission`; do not promote it.

## 5. Rebuild the leaderboard

```bash
uv run mnnc collect --replay-dir replays
uv run mnnc render leaderboard.json docs --replays replays
```

`collect` re-evaluates every model and is the authoritative rebuild. It also
records a seed-0 replay for each Snake submission. `replays/` is temporary;
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
