from __future__ import annotations

import json
from pathlib import Path

from minimal_nn_competition.submissions import github
from minimal_nn_competition.submissions.workflow import WorkflowError, _score_candidate


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


def test_merge_pins_the_validated_head(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(github, "_gh", lambda args: calls.append(args) or "")

    github.merge(3, delete_branch=False, head_sha="a" * 40)

    assert calls == [
        ["pr", "merge", "3", "--merge", "--match-head-commit", "a" * 40],
    ]


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
            "a" * 40,
            1,
            Path("models/mnist/partial"),
            tmp_path,
            tmp_path / "data",
        )
        is result
    )


def test_run_requires_submission_branch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "minimal_nn_competition.submissions.workflow._git",
        lambda args, cwd: "dev\n" if args == ["branch", "--show-current"] else "",
    )

    from minimal_nn_competition.submissions.workflow import run

    try:
        run(tmp_path, None, Path("data"))
    except WorkflowError as exc:
        assert "submission branch" in str(exc)
    else:
        raise AssertionError("run should reject non-submission branches")
