from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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
        "minimal_nn_competition.submissions.workflow.score_submission_process",
        lambda worktree, directory, data_root, timeout=600: result,
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


def test_score_submission_process_success(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from minimal_nn_competition.submissions.scoring import score_submission_process

    result = score_submission_process(root, Path("models/snake-simple-8x8/example"), tmp_path)
    assert result.passed is not None


def test_score_submission_process_timeout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.scoring import score_submission_process

    with pytest.raises(SubmissionError, match="timed out"):
        score_submission_process(
            root,
            Path("models/snake-simple-8x8/example"),
            tmp_path,
            timeout=0.001,
        )


def test_score_submission_process_failure(tmp_path: Path) -> None:
    import json

    import torch

    root = tmp_path / "repo"
    root.mkdir()
    (root / "models" / "snake-simple-8x8" / "bad").mkdir(parents=True)
    (root / "models" / "snake-simple-8x8" / "bad" / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "snake-simple-8x8",
                "name": "bad",
                "author": "tester",
                "architecture": "linear",
                "input_size": 24,
                "output_size": 4,
            }
        )
    )
    torch.save(
        {"weight": torch.zeros((3, 7)), "bias": torch.zeros(3)},
        root / "models" / "snake-simple-8x8" / "bad" / "model.pt",
    )
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.scoring import score_submission_process

    with pytest.raises(SubmissionError):
        score_submission_process(root, Path("models/snake-simple-8x8/bad"), tmp_path)


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS is not enforceable on macOS")
def test_score_submission_process_resource_exhaustion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.scoring import score_submission_process

    with pytest.raises(SubmissionError):
        score_submission_process(
            root,
            Path("models/snake-simple-8x8/example"),
            tmp_path,
            memory_limit_bytes=1,
        )
