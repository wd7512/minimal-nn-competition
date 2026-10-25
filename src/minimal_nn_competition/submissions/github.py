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
            "number,title,headRefName,headRefOid,baseRefName,statusCheckRollup",
        ]
    )
    prs = []
    for item in json.loads(raw):
        checks = item.get("statusCheckRollup", [])
        conclusions = [check.get("conclusion") for check in checks if check.get("conclusion")]
        passed = bool(conclusions) and all(conclusion == "SUCCESS" for conclusion in conclusions)
        prs.append(
            PullRequest(
                item["number"],
                item["title"],
                item["headRefOid"],
                item["baseRefName"],
                item["headRefName"],
                passed,
            )
        )
    return sorted(prs, key=lambda pr: pr.number)


def comment(number: int, body: str) -> None:
    _gh(["pr", "comment", str(number), "--body", body])


def merge(number: int, *, delete_branch: bool) -> None:
    args = ["pr", "merge", str(number), "--merge"]
    if delete_branch:
        args.append("--delete-branch")
    _gh(args)


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
