"""Small, testable wrapper around the GitHub CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


class GitHubError(RuntimeError):
    """Raised for an unavailable or failed GitHub CLI operation."""


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    head_sha: str
    base_ref: str
    head_ref: str
    checks_passed: bool
    checks_summary: str = ""
    checks_status: str = "none"
    state: str = "OPEN"
    merged: bool = False
    merge_commit_sha: str | None = None


def _gh(args: list[str], *, check: bool = True) -> str:
    if shutil.which("gh") is None:
        raise GitHubError("GitHub CLI is required; install `gh` and run `gh auth login`.")
    try:
        result = subprocess.run(["gh", *args], check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        if "not logged" in detail.lower() or "authentication" in detail.lower():
            detail = "run `gh auth login`"
        raise GitHubError(f"GitHub CLI request failed: {detail or 'unknown error'}") from exc
    return result.stdout


def _checks(item: dict) -> tuple[bool, str, str]:
    """Return (passed, status, summary); status is none/pending/failed/passed."""
    checks = item.get("statusCheckRollup") or []
    if not checks:
        return False, "none", "no GitHub checks have run"

    # A successful check alongside a pending check is still pending. Each item
    # must have an explicit successful conclusion/state before the PR is eligible.
    passed = all(
        check.get("conclusion") == "SUCCESS"
        if check.get("conclusion") is not None
        else check.get("state") == "SUCCESS"
        for check in checks
    )
    if passed:
        return True, "passed", "all GitHub checks passed"

    failed = []
    for check in checks:
        conclusion = check.get("conclusion")
        state = check.get("state")
        settled_failure = conclusion is not None and conclusion != "SUCCESS"
        unexpected_state = conclusion is None and state not in (
            None,
            "SUCCESS",
            "PENDING",
            "IN_PROGRESS",
            "QUEUED",
            "EXPECTED",
        )
        if settled_failure or unexpected_state:
            failed.append(check.get("name", "unnamed check"))
    if failed:
        return False, "failed", "failed checks: " + ", ".join(failed)
    return False, "pending", "GitHub checks are still pending"


def _pull_request(item: dict) -> PullRequest:
    checks_passed, checks_status, checks_summary = _checks(item)
    state = str(item.get("state", "OPEN")).upper()
    merged = state == "MERGED" or bool(item.get("mergedAt"))
    merge_commit = item.get("mergeCommit") or {}
    merge_commit_sha = merge_commit.get("oid") if isinstance(merge_commit, dict) else None
    return PullRequest(
        number=int(item["number"]),
        title=str(item.get("title", "")),
        head_sha=str(item["headRefOid"]),
        base_ref=str(item["baseRefName"]),
        head_ref=str(item["headRefName"]),
        checks_passed=checks_passed,
        checks_summary=checks_summary,
        checks_status=checks_status,
        state=state,
        merged=merged,
        merge_commit_sha=merge_commit_sha,
    )


def check_authentication() -> None:
    _gh(["auth", "status"])


def open_submission_prs() -> list[PullRequest]:
    raw = _gh(
        [
            "pr",
            "list",
            "--base",
            "submission",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,headRefName,headRefOid,baseRefName,statusCheckRollup,state,mergedAt,mergeCommit",
        ]
    )
    return sorted((_pull_request(item) for item in json.loads(raw)), key=lambda pr: pr.number)


def get_pull_request(number: int) -> PullRequest:
    """Read a PR even if already merged, enabling recovery of post-merge failures."""
    raw = _gh(
        [
            "pr",
            "view",
            str(number),
            "--json",
            "number,title,headRefName,headRefOid,baseRefName,statusCheckRollup,state,mergedAt,mergeCommit",
        ]
    )
    return _pull_request(json.loads(raw))


def comment(number: int, body: str) -> None:
    _gh(["pr", "comment", str(number), "--body", body])


def merge(number: int, *, delete_branch: bool) -> None:
    args = ["pr", "merge", str(number), "--merge"]
    if delete_branch:
        args.append("--delete-branch")
    _gh(args)


def delete_remote_branch(number: int) -> None:
    """Best-effort delete of a merged PR's head branch, only if it is in this repo."""
    raw = _gh(
        [
            "pr",
            "view",
            str(number),
            "--json",
            "headRefName,headRepository,headRepositoryOwner",
        ]
    )
    item = json.loads(raw)
    repository = item.get("headRepository") or {}
    owner = item.get("headRepositoryOwner") or {}
    branch = item.get("headRefName")
    repo_name = repository.get("name") if isinstance(repository, dict) else None
    owner_name = owner.get("login") if isinstance(owner, dict) else None
    if (
        not branch
        or not branch.startswith("submit/")
        or branch in {"main", "dev", "submission"}
        or not repo_name
        or not owner_name
    ):
        return
    base = json.loads(_gh(["repo", "view", "--json", "nameWithOwner"]))
    if f"{owner_name}/{repo_name}" != base.get("nameWithOwner"):
        return
    # Deleting the remote ref is separate from merging: it cannot trigger gh's
    # local-branch cleanup, which can fail when that branch is checked out in a worktree.
    _gh(
        [
            "api",
            "--method",
            "DELETE",
            f"repos/{owner_name}/{repo_name}/git/refs/heads/{branch}",
        ],
        check=False,
    )


def find_promotion_pr() -> int | None:
    """Return an already-open submission-to-main promotion PR, if one exists."""
    raw = _gh(
        [
            "pr",
            "list",
            "--base",
            "main",
            "--head",
            "submission",
            "--state",
            "open",
            "--limit",
            "1",
            "--json",
            "number",
        ]
    )
    items = json.loads(raw)
    return int(items[0]["number"]) if items else None


def create_promotion_pr() -> int:
    raw = _gh(
        [
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            "submission",
            "--title",
            "Promote submission leaderboard updates",
            "--body",
            "Automated submission promotion.",
        ]
    )
    return int(raw.rstrip().split("/")[-1])


def merge_promotion(number: int) -> None:
    merge(number, delete_branch=False)
