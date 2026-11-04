#!/usr/bin/env bash
# Vibe submission builder CI script.
# Env: VIBE_SMOKE ("1"/"0"), VIBE_NAME, VIBE_TARGET (branch to push submission to)
set -uo pipefail

LOG=/tmp/vibe-run.log
: > "$LOG"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

publish() {
  local rc=0
  echo "[$(date -u +%H:%M:%S)] publishing run log via gh api" >> "$LOG" 2>&1
  B64=$(base64 -w0 "$LOG")
  ARGS=(-X PUT -H "Accept: application/vnd.github+json"
        -f message="Update vibe run log"
        -f branch=vibe-builder
        -f content="$B64")
  SHA=$(gh api "repos/wd7512/minimal-nn-competition/contents/vibe/last-run.log?ref=vibe-builder" --jq .sha 2>>"$LOG" || true)
  if [ -n "${SHA:-}" ]; then ARGS+=(-f sha="$SHA"); fi
  gh api "repos/wd7512/minimal-nn-competition/contents/vibe/last-run.log" "${ARGS[@]}" >>"$LOG" 2>&1 || { echo "gh api publish FAILED" >> "$LOG"; rc=1; }
  return $rc
}
trap 'publish' EXIT

fail() { log "FAILED at step: $1"; log "last 40 log lines:"; tail -n 40 "$LOG" | while IFS= read -r l; do echo "  $l" >> "$LOG"; done; log "END (failure)"; exit 1; }

log "=== Vibe CI start ==="
log "smoke=${VIBE_SMOKE:-unset} name=${VIBE_NAME:-unset} target=${VIBE_TARGET:-unset}"
log "runner: $(uname -a)"
log "pwd=$(pwd)"
log "git head: $(git rev-parse HEAD 2>>"$LOG") branch checkout ref"

log "step 1/6: uv sync"
uv sync --dev --locked >>"$LOG" 2>&1 || fail "uv sync"

log "step 2/6: build submission (builder/vibe/build.py)"
uv run --no-sync python builder/vibe/build.py >>"$LOG" 2>&1 || fail "build.py"

log "step 3/6: official evaluation"
uv run --no-sync mnnc evaluate snake-simple-16x16 "models/snake-simple-16x16/${VIBE_NAME}" >>"$LOG" 2>&1 || fail "evaluate"

log "step 4/6: validate submission gate"
uv run --no-sync mnnc validate-submission "models/snake-simple-16x16/${VIBE_NAME}" >>"$LOG" 2>&1 || fail "validate-submission"

log "step 5/6: commit submission files"
git config user.name "GLM x le vibe"
git config user.email "wd7512@users.noreply.github.com"
git add "models/snake-simple-16x16/${VIBE_NAME}"
git commit -m "Add snake-simple-16x16/${VIBE_NAME}" >>"$LOG" 2>&1 || fail "git commit"

log "step 6/6: push to ${VIBE_TARGET}"
git push --force origin "HEAD:${VIBE_TARGET}" >>"$LOG" 2>&1 || fail "git push"

log "=== SUCCESS ==="
