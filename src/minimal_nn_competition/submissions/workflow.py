"""Sequential submission acceptance workflow."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from minimal_nn_competition.benchmarks.snake.replay import load_replay
from minimal_nn_competition.leaderboard import load_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.submissions.github import (
    GitHubError,
    PullRequest,
    check_authentication,
    comment,
    create_promotion_pr,
    merge,
    merge_promotion,
    open_submission_prs,
)
from minimal_nn_competition.submissions.scoring import score_submission, upsert_result
from minimal_nn_competition.submissions.validation import submission_directory, validate_diff

logger = logging.getLogger(__name__)


class WorkflowError(RuntimeError):
    """Raised when the acceptance workflow cannot safely continue."""


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"git {' '.join(args)} failed: {(exc.stderr or '').strip()}") from exc


def _validate_pr(pr: PullRequest, root: Path) -> Path:
    if pr.base_ref != "submission":
        raise WorkflowError("PR does not target submission")
    if not pr.checks_passed:
        raise WorkflowError(f"required GitHub checks have not passed ({pr.checks_summary})")
    diff = _git(["diff", "--name-status", f"origin/submission...{pr.head_sha}"], root)
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


def _load_replays(root: Path) -> dict:
    replay_dir = root / "docs" / "games"
    if not replay_dir.is_dir():
        return {}
    return {path.stem: load_replay(path) for path in sorted(replay_dir.glob("*.json"))}


def run(root: Path, pr_number: int | None, data_root: Path) -> int:
    branch = _git(["branch", "--show-current"], root).strip()
    if branch != "submission":
        raise WorkflowError(f"run from the submission branch, not {branch or 'detached HEAD'}")
    check_authentication()
    prs = open_submission_prs()
    if pr_number is not None:
        prs = [pr for pr in prs if pr.number == pr_number]
        if not prs:
            raise WorkflowError(f"open PR {pr_number} targeting submission was not found")
    if not prs:
        logger.info("No open submission PRs found")
        return 0

    _git(["fetch", "origin", "main", "submission"], root)
    accepted = 0
    skipped = 0
    for pr in prs:
        try:
            _git(
                [
                    "fetch",
                    "origin",
                    f"pull/{pr.number}/head:refs/remotes/origin/pr/{pr.number}",
                ],
                root,
            )
            directory = _validate_pr(pr, root)
            result = _score_candidate(pr, directory, root, root / data_root)
            merge(pr.number)
            _git(["pull", "--ff-only", "origin", "submission"], root)
            upsert_result(root / "leaderboard.json", result)
            render_site(load_results(root / "leaderboard.json"), root / "docs", _load_replays(root))
            _git(["add", "leaderboard.json", "docs"], root)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
                _git(["commit", "-m", f"Score submission PR #{pr.number}"], root)
            _git(["push", "origin", "submission"], root)
            accepted += 1
            logger.info("Accepted PR #%d (%s)", pr.number, result.submission)
        except (GitHubError, OSError, RuntimeError, ValueError, TypeError) as exc:
            skipped += 1
            message = f"Automated submission validation failed:\n\n{exc}"
            logger.error("PR #%d skipped: %s", pr.number, exc)
            comment(pr.number, message)

    if not accepted:
        logger.info(
            "Validation summary: %d processed, %d accepted, %d skipped", len(prs), accepted, skipped
        )
        logger.info("No submissions accepted; skipping promotion")
        return 0
    promotion = create_promotion_pr()
    merge_promotion(promotion)
    _git(["fetch", "origin", "main"], root)
    _git(["merge", "--ff-only", "origin/main"], root)
    _git(["push", "origin", "submission"], root)
    logger.info(
        "Validation summary: %d processed, %d accepted, %d skipped", len(prs), accepted, skipped
    )
    logger.info("Promoted accepted submissions to main")
    return 0
