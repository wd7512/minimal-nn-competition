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

`--delete-branch` deletes the PR head branch. It is safe here because the head
is a short-lived `submit/*` branch; never use it for a promotion PR whose head
is `submission` or `dev`.

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

Expected scores: `1.0` everywhere — every seed or sample must pass. MNIST and
Fashion-MNIST classify all 70,000 examples; Chess places every position within
0.1 pawns of Stockfish; Snake fills the board on every seed; CartPole survives
500 steps on every seed; Pendulum clears `-500` per seed; LunarLander lands
every seed. The checked-in chess-eval-100k example is the one expected
failure: it fails the larger tier on purpose. Confirm the score and parameter
count match the PR description, and that an image cache is the correct
benchmark's data.

Snake Obstacles 8x8 passes only when every seed fills its free cells and
completes the extra loop move. Inspect `seed_passed` and `seed_target_lengths` in
the result, and validate the layouts separately:

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
records a seed-0 replay for each Snake and control submission. `replays/` is
temporary;
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

Wait for CI, then merge without `--delete-branch`:

```bash
gh pr merge <number> --merge
git fetch origin
git merge --ff-only origin/main
git push origin submission
```

Confirm the Pages deployment succeeds and the new rows appear at
<https://wd7512.github.io/minimal-nn-competition/>.
