"""Sequential submission acceptance workflow."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from minimal_nn_competition.benchmarks.control.names import RL_BENCHMARKS
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import selected_input_indices
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.benchmarks.snake.replay import (
    REPLAY_SEED as SNAKE_REPLAY_SEED,
)
from minimal_nn_competition.benchmarks.snake.replay import (
    load_replay,
    replay_stem,
    save_replay,
)
from minimal_nn_competition.benchmarks.snake.replay import (
    record_game as record_snake_game,
)
from minimal_nn_competition.leaderboard import load_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.submissions.github import (
    GitHubError,
    PullRequest,
    check_authentication,
    comment,
    create_promotion_pr,
    delete_remote_branch,
    get_pull_request,
    merge,
    merge_promotion,
    open_submission_prs,
)
from minimal_nn_competition.submissions.scoring import result_key, score_submission, upsert_result
from minimal_nn_competition.submissions.validation import submission_directory, validate_diff

logger = logging.getLogger(__name__)
_RECOVERY_ERRORS = (GitHubError, OSError, RuntimeError, ValueError, TypeError, KeyError)


class WorkflowError(RuntimeError):
    """Raised when the acceptance workflow cannot safely continue."""


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"git {' '.join(args)} failed: {(exc.stderr or '').strip()}") from exc


def _is_ancestor(ancestor: str, descendant: str, root: Path) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        detail = (result.stderr or result.stdout).strip()
        raise WorkflowError(f"could not compare {ancestor} and {descendant}: {detail}")
    return result.returncode == 0


def _ledger_keys_from_payload(payload: str, source: str) -> set[tuple[str, str]]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"could not parse leaderboard from {source}: {exc}") from exc
    if not isinstance(rows, list):
        raise WorkflowError(f"leaderboard from {source} must be a JSON list")
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("benchmark") or not row.get("submission"):
            raise WorkflowError(f"leaderboard from {source} contains an invalid row")
        keys.add((str(row["benchmark"]), str(row["submission"])))
    return keys


def _local_ledger_keys(root: Path) -> set[tuple[str, str]]:
    path = root / "docs" / "leaderboard.json"
    if not path.exists():
        return set()
    return {result_key(result) for result in load_results(path)}


def _remote_ledger_keys(root: Path, ref: str = "origin/submission") -> set[tuple[str, str]]:
    payload = _git(["show", f"{ref}:docs/leaderboard.json"], root)
    return _ledger_keys_from_payload(payload, ref)


def _sync_submission(root: Path) -> None:
    """Fetch and fast-forward from the integration branch without reconciling histories."""
    _git(["fetch", "origin", "main", "submission"], root)
    if not _is_ancestor("origin/main", "origin/submission", root):
        raise WorkflowError(
            "origin/submission does not contain origin/main. Promote or reconcile the branches "
            "using the normal PR flow before accepting submissions; the workflow will not merge "
            "main into submission or rewrite either branch during acceptance."
        )

    if _is_ancestor("origin/submission", "submission", root):
        # Local submission is current or contains local commits that still need pushing.
        pass
    elif _is_ancestor("submission", "origin/submission", root):
        _git(["merge", "--ff-only", "origin/submission"], root)
    else:
        raise WorkflowError(
            "local submission and origin/submission have diverged. Reconcile them explicitly; "
            "the acceptance workflow will not reset, rebase, or force-push submission."
        )

    # A local commit left behind by a previous run must not be allowed to erase rows
    # present at the current remote tip.
    local_keys = _local_ledger_keys(root)
    remote_keys = _remote_ledger_keys(root)
    if not remote_keys.issubset(local_keys):
        missing = sorted(remote_keys - local_keys)
        raise WorkflowError(
            "local leaderboard is missing entries from origin/submission; refusing to push "
            f"a possibly stale ledger. Missing keys: {missing[:8]}"
        )


def _validate_pr(pr: PullRequest, root: Path, validation_base: str | None = None) -> Path:
    if pr.base_ref != "submission":
        raise WorkflowError("PR does not target submission")
    if not pr.merged and not pr.checks_passed:
        raise WorkflowError(f"required GitHub checks have not passed ({pr.checks_summary})")

    if pr.merged:
        if not pr.merge_commit_sha:
            raise WorkflowError(
                "PR is merged but GitHub did not return its merge commit; rerun once metadata is available"
            )
        # For a merged PR, compare the merge commit to its first parent. Comparing the
        # head to the now-advanced integration branch would produce an empty diff.
        diff = _git(
            ["diff", "--name-status", f"{pr.merge_commit_sha}^1", pr.merge_commit_sha], root
        )
    else:
        # A three-dot diff is only safe when the current integration tip is the PR base
        # (or its ancestor). Otherwise unrelated files can leak into the candidate diff.
        base = validation_base or "origin/submission"
        if not _is_ancestor(base, pr.head_sha, root):
            raise WorkflowError(
                f"PR head is not based on the validation base {base}. Rebase the submission "
                "branch onto the current origin/submission tip, then rerun validation."
            )
        diff = _git(["diff", "--name-status", f"{base}...{pr.head_sha}"], root)

    with tempfile.TemporaryDirectory(prefix=f"mnnc-validate-{pr.number}-") as temp:
        worktree = Path(temp) / "tree"
        _git(["worktree", "add", "--detach", str(worktree), pr.head_sha], root)
        try:
            errors = validate_diff(diff, worktree)
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], root)
        if errors:
            raise WorkflowError("\n".join(errors))
    return submission_directory(diff)


def _score_candidate(pr: PullRequest, directory: Path, root: Path, data_root: Path):
    with tempfile.TemporaryDirectory(prefix=f"mnnc-pr-{pr.number}-") as temp:
        worktree = Path(temp) / "tree"
        _git(["worktree", "add", "--detach", str(worktree), pr.head_sha], root)
        try:
            result = score_submission(worktree, directory, data_root)
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], root)
    return result


def _record_submission_replay(root: Path, directory: Path) -> Path | None:
    """Write the deterministic detail-page replay for an accepted game submission."""
    submission = Submission.from_directory(root / directory)
    benchmark_name = submission.config.benchmark
    if benchmark_name not in SNAKE_BENCHMARKS + RL_BENCHMARKS:
        return None

    benchmark = create_benchmark(benchmark_name)
    loaded = benchmark.load_submission(submission)
    name = directory.name
    replay_dir = root / "docs" / "games"
    stem = replay_stem(benchmark_name, name)

    if benchmark_name in SNAKE_BENCHMARKS:
        payload = record_snake_game(
            loaded.model,
            seed=SNAKE_REPLAY_SEED,
            max_moves=benchmark.max_moves,
            input_indices=selected_input_indices(submission.config, benchmark_name),
            **benchmark.environment_options(SNAKE_REPLAY_SEED),
        )
    else:
        from minimal_nn_competition.benchmarks.control.replay import record_game

        payload = record_game(benchmark_name, loaded.model, steps=benchmark.max_steps)

    return save_replay(replay_dir / f"{stem}.json", payload)


def _load_replays(root: Path) -> dict:
    replay_dir = root / "docs" / "games"
    if not replay_dir.is_dir():
        return {}
    return {path.stem: load_replay(path) for path in sorted(replay_dir.glob("*.json"))}


def _publish_result(root: Path, directory: Path, result) -> None:
    """Upsert and publish artifacts, rebasing the generated commit on concurrent writes."""
    ledger_path = root / "docs" / "leaderboard.json"

    for attempt in range(3):
        base_sha = _git(["rev-parse", "HEAD"], root).strip()
        old_keys = _local_ledger_keys(root)
        remote_keys = _remote_ledger_keys(root)
        if not remote_keys.issubset(old_keys):
            raise WorkflowError(
                "refusing leaderboard update: local ledger is missing entries from "
                "origin/submission"
            )

        upsert_result(ledger_path, result)
        new_keys = _local_ledger_keys(root)
        if not old_keys.issubset(new_keys):
            raise WorkflowError("leaderboard upsert removed an existing entry")
        if result_key(result) not in new_keys:
            raise WorkflowError("leaderboard upsert did not record the scored submission")

        _record_submission_replay(root, directory)
        render_site(load_results(ledger_path), root / "docs", _load_replays(root))
        _git(["add", "docs"], root)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False)
        if staged.returncode == 1:
            _git(["commit", "-m", f"Score submission {result.submission}"], root)
        elif staged.returncode != 0:
            raise WorkflowError("could not inspect staged leaderboard changes")

        try:
            _git(["push", "origin", "submission"], root)
        except WorkflowError as push_error:
            _git(["fetch", "origin", "submission"], root)
            remote_tip = _git(["rev-parse", "origin/submission"], root).strip()
            if remote_tip == base_sha:
                # The push failed without a competing write (e.g. permission failure).
                # Keep the local score commit so an explicit recovery run can retry it.
                raise push_error
            if not _is_ancestor(base_sha, remote_tip, root):
                raise WorkflowError(
                    "push failed and origin/submission did not advance from the publish base; "
                    "refusing to reset potentially unrelated local commits"
                ) from push_error
            # Another run advanced the integration branch after our ledger read. This
            # attempt's commit is workflow-generated and is the only commit after base_sha.
            # Drop it, fast-forward to the remote tip, and rebuild the union ledger from there.
            _git(["reset", "--hard", "origin/submission"], root)
            if attempt == 2:
                raise WorkflowError(
                    "origin/submission kept advancing during publication; the local branch "
                    "was reset to the remote tip safely. Rerun validation to publish the score."
                ) from push_error
            continue

        _git(["fetch", "origin", "submission"], root)
        pushed_keys = _remote_ledger_keys(root)
        if not old_keys.issubset(pushed_keys):
            raise WorkflowError(
                "remote leaderboard read-back is missing previously recorded entries"
            )
        if result_key(result) not in pushed_keys:
            raise WorkflowError("remote leaderboard read-back is missing the new submission")
        return

    raise WorkflowError("could not publish leaderboard after repeated concurrent updates")


def _refresh_missing_checks(pr: PullRequest) -> PullRequest:
    """Give delayed GitHub check events a short bounded window to appear."""
    if pr.merged or pr.checks_summary != "no GitHub checks have run":
        return pr
    for delay in (1, 2):
        time.sleep(delay)
        pr = get_pull_request(pr.number)
        if pr.merged or pr.checks_summary != "no GitHub checks have run":
            return pr
    return pr


def _is_merged_now(pr: PullRequest) -> PullRequest:
    if pr.merged:
        return pr
    try:
        latest = get_pull_request(pr.number)
    except GitHubError:
        return pr
    return latest if latest.merged else pr


def run(root: Path, pr_number: int | None, data_root: Path) -> int:
    branch = _git(["branch", "--show-current"], root).strip()
    if branch != "submission":
        raise WorkflowError(f"run from the submission branch, not {branch or 'detached HEAD'}")
    if _git(["status", "--porcelain", "--untracked-files=no"], root).strip():
        raise WorkflowError(
            "working tree has uncommitted tracked changes; commit or stash them first so the "
            "acceptance workflow cannot accidentally carry them into the leaderboard commit"
        )

    check_authentication()
    _sync_submission(root)
    validation_base = _git(["rev-parse", "origin/submission"], root).strip()
    prs = open_submission_prs()
    if pr_number is not None:
        prs = [pr for pr in prs if pr.number == pr_number]
        if not prs:
            # Merged PRs are deliberately not returned by `gh pr list --state open`.
            # Allow an explicit rerun to recover a merge whose ledger step failed.
            recovered_pr = get_pull_request(pr_number)
            if recovered_pr.base_ref != "submission" or not recovered_pr.merged:
                raise WorkflowError(
                    f"open PR {pr_number} targeting submission was not found; it is not a merged PR "
                    "that can be recovered"
                )
            prs = [recovered_pr]
    if not prs:
        logger.info("No open submission PRs found")
        return 0

    accepted = 0
    skipped = 0
    for listed_pr in prs:
        pr = listed_pr
        directory = None
        result = None
        merged = pr.merged
        try:
            if not pr.merged and pr.checks_summary == "no GitHub checks have run":
                pr = _refresh_missing_checks(pr)
            if not pr.merged and not pr.checks_passed:
                raise WorkflowError(
                    f"required GitHub checks are not successful ({pr.checks_summary}); "
                    "rerun once checks have completed"
                )

            _git(["fetch", "origin", "submission"], root)
            _git(
                [
                    "fetch",
                    "origin",
                    f"pull/{pr.number}/head:refs/remotes/origin/pr/{pr.number}",
                ],
                root,
            )
            directory = _validate_pr(pr, root, validation_base)
            result = _score_candidate(pr, directory, root, root / data_root)

            if not pr.merged:
                # Do not use `--delete-branch`: gh may fail deleting a local branch that
                # is checked out in another worktree after the remote merge has succeeded.
                merge(pr.number, delete_branch=False)
                merged = True
            _git(["pull", "--ff-only", "origin", "submission"], root)
            _publish_result(root, directory, result)
            accepted += 1
            logger.info("Accepted PR #%d (%s)", pr.number, result.submission)

            try:
                delete_remote_branch(pr.number)
            except _RECOVERY_ERRORS as cleanup_error:
                # Branch cleanup is housekeeping, not part of accepting/scoring a model.
                logger.warning(
                    "Could not clean remote head branch for PR #%d: %s", pr.number, cleanup_error
                )
        except _RECOVERY_ERRORS as exc:
            current_pr = _is_merged_now(pr)
            merged = merged or current_pr.merged
            if merged:
                # A merged PR is never reported as an unaccepted submission. Try to finish
                # scoring/artifacts now; an explicit `--pr N` rerun can resume if this fails.
                try:
                    if directory is None:
                        directory = _validate_pr(current_pr, root, validation_base)
                    if result is None:
                        result = _score_candidate(current_pr, directory, root, root / data_root)
                    _sync_submission(root)
                    _git(["pull", "--ff-only", "origin", "submission"], root)
                    _publish_result(root, directory, result)
                    accepted += 1
                    logger.warning("Recovered post-merge acceptance for PR #%d", pr.number)
                    continue
                except _RECOVERY_ERRORS as recovery_error:
                    skipped += 1
                    logger.error(
                        "PR #%d is already merged but scoring/ledger recovery still failed: %s. "
                        "Rerun `mnnc validate-submissions --pr %d` after fixing the cause.",
                        pr.number,
                        recovery_error,
                        pr.number,
                    )
                    continue

            skipped += 1
            message = f"Automated submission validation failed:\n\n{exc}"
            logger.error("PR #%d skipped: %s", pr.number, exc)
            comment(pr.number, message)

    logger.info(
        "Validation summary: %d processed, %d accepted, %d skipped", len(prs), accepted, skipped
    )
    if not accepted:
        logger.info("No submissions accepted; skipping promotion")
        return 0

    _git(["fetch", "origin", "main", "submission"], root)
    ahead = int(_git(["rev-list", "--count", "origin/main..origin/submission"], root).strip())
    if ahead == 0:
        logger.info("All scored artifacts are already promoted to main")
        return 0
    promotion = create_promotion_pr()
    merge_promotion(promotion)
    _git(["fetch", "origin", "main"], root)
    _git(["merge", "--ff-only", "origin/main"], root)
    _git(["push", "origin", "submission"], root)
    logger.info("Promoted accepted submissions to main")
    return 0
