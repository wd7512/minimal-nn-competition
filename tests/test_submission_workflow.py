from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimal_nn_competition.submissions import github
from minimal_nn_competition.submissions.github import PullRequest
from minimal_nn_competition.submissions.workflow import _score_candidate


def test_open_submission_prs_requires_successful_check(monkeypatch) -> None:
    payload = [
        {
            "number": 2,
            "title": "pending",
            "headRefName": "pending",
            "headRefOid": "b" * 40,
            "baseRefName": "submission",
            "statusCheckRollup": [],
        },
        {
            "number": 1,
            "title": "passed",
            "headRefName": "passed",
            "headRefOid": "a" * 40,
            "baseRefName": "submission",
            "statusCheckRollup": [{"conclusion": "SUCCESS"}],
        },
    ]
    monkeypatch.setattr(github, "_gh", lambda args: json.dumps(payload))

    prs = github.open_submission_prs()

    assert [pr.number for pr in prs] == [1, 2]
    assert prs[0].checks_passed is True
    assert prs[1].checks_passed is False
    assert prs[1].checks_summary == "no GitHub checks have run"


def test_merge_deletes_short_lived_submission_branch(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(github, "_gh", lambda args: calls.append(args) or "")

    github.merge(1, delete_branch=True)

    assert calls == [["pr", "merge", "1", "--merge", "--delete-branch"]]


def test_merge_promotion_keeps_long_lived_branch(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(github, "_gh", lambda args: calls.append(args) or "")

    github.merge_promotion(2)

    assert calls == [["pr", "merge", "2", "--merge"]]


def test_open_submission_prs_rejects_failed_check(monkeypatch) -> None:
    payload = [
        {
            "number": 1,
            "title": "failed",
            "headRefName": "failed",
            "headRefOid": "a" * 40,
            "baseRefName": "submission",
            "statusCheckRollup": [
                {"name": "validate", "conclusion": "SUCCESS"},
                {"name": "validate", "conclusion": "FAILURE"},
            ],
        }
    ]
    monkeypatch.setattr(github, "_gh", lambda args: json.dumps(payload))

    assert github.open_submission_prs()[0].checks_passed is False
    assert github.open_submission_prs()[0].checks_summary == "failed checks: validate"


def test_score_candidate_accepts_partial_result(monkeypatch, tmp_path: Path) -> None:
    result = type("Result", (), {"passed": False, "score": 0.5})()
    monkeypatch.setattr(
        "minimal_nn_competition.submissions.workflow._git",
        lambda args, cwd: str(tmp_path),
    )
    monkeypatch.setattr(
        "minimal_nn_competition.submissions.workflow.score_submission",
        lambda root, directory, data_root: result,
    )

    assert (
        _score_candidate(
            PullRequest(1, "partial", "a" * 40, "submission", "branch", True),
            Path("models/mnist/partial"),
            tmp_path,
            tmp_path / "data",
        )
        is result
    )


def test_run_checks_out_submission_and_updates_with_main(monkeypatch, tmp_path: Path) -> None:
    from minimal_nn_competition.submissions import workflow

    calls: list[list[str]] = []

    def fake_git(args: list[str], cwd: Path) -> str:
        calls.append(args)
        if args == ["branch", "--show-current"]:
            return "main\n"
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [])

    assert workflow.run(tmp_path, None, Path("data")) == 0
    assert ["checkout", "submission"] in calls
    assert ["merge", "--ff-only", "origin/main"] in calls


def test_run_skips_checkout_on_submission_branch(monkeypatch, tmp_path: Path) -> None:
    from minimal_nn_competition.submissions import workflow

    calls: list[list[str]] = []

    def fake_git(args: list[str], cwd: Path) -> str:
        calls.append(args)
        if args == ["branch", "--show-current"]:
            return "submission\n"
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [])

    assert workflow.run(tmp_path, None, Path("data")) == 0
    assert ["checkout", "submission"] not in calls
    assert ["merge", "--ff-only", "origin/main"] in calls


def test_run_rejects_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    from minimal_nn_competition.submissions import workflow

    def fake_git(args: list[str], cwd: Path) -> str:
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return " M src/main.py\n"
        if args == ["branch", "--show-current"]:
            return "main\n"
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [])

    with pytest.raises(workflow.WorkflowError, match="uncommitted changes"):
        workflow.run(tmp_path, None, Path("data"))
