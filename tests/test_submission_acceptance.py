from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions import github, scoring, workflow
from minimal_nn_competition.submissions.github import PullRequest


def _result(submission: str, benchmark: str = "mnist") -> BenchmarkResult:
    return BenchmarkResult(
        submission=submission,
        benchmark=benchmark,
        score=1.0,
        passed=True,
        parameter_count=1,
        details={},
        name=submission,
        author="test",
    )


def test_upsert_preserves_same_submission_name_in_other_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.json"
    scoring.save_results(path, [_result("example", "mnist"), _result("example", "cartpole")])

    scoring.upsert_result(path, _result("example", "mnist"))

    keys = {(item.benchmark, item.submission) for item in scoring.load_results(path)}
    assert keys == {("mnist", "example"), ("cartpole", "example")}


def test_check_rollup_does_not_pass_while_a_check_is_pending() -> None:
    passed, status, summary = github._checks(
        {
            "statusCheckRollup": [
                {"name": "lint", "conclusion": "SUCCESS"},
                {"name": "tests", "conclusion": None, "state": "IN_PROGRESS"},
            ]
        }
    )

    assert passed is False
    assert status == "pending"
    assert "pending" in summary


def test_check_rollup_reports_structured_status() -> None:
    assert github._checks({}) == (False, "none", "no GitHub checks have run")
    assert github._checks({"statusCheckRollup": [{"name": "lint", "conclusion": "SUCCESS"}]}) == (
        True,
        "passed",
        "all GitHub checks passed",
    )
    assert github._checks({"statusCheckRollup": [{"name": "lint", "conclusion": "FAILURE"}]}) == (
        False,
        "failed",
        "failed checks: lint",
    )


def test_pr_parser_retains_merged_metadata() -> None:
    pr = github._pull_request(
        {
            "number": 12,
            "title": "already merged",
            "headRefName": "submit/model",
            "headRefOid": "a" * 40,
            "baseRefName": "submission",
            "state": "MERGED",
            "mergedAt": "2026-10-26T00:00:00Z",
            "mergeCommit": {"oid": "b" * 40},
            "statusCheckRollup": [{"conclusion": "SUCCESS"}],
        }
    )

    assert pr.merged is True
    assert pr.merge_commit_sha == "b" * 40


def test_open_pr_requires_current_validation_base(monkeypatch, tmp_path: Path) -> None:
    pr = PullRequest(1, "stale", "a" * 40, "submission", "branch", True)
    monkeypatch.setattr(workflow, "_is_ancestor", lambda ancestor, descendant, root: False)

    with pytest.raises(workflow.WorkflowError, match="not based on the validation base"):
        workflow._validate_pr(pr, tmp_path, "c" * 40)


def test_ledger_payload_keys_are_scoped_by_benchmark() -> None:
    keys = workflow._ledger_keys_from_payload(
        json.dumps(
            [
                {"benchmark": "mnist", "submission": "example"},
                {"benchmark": "cartpole", "submission": "example"},
            ]
        ),
        "test fixture",
    )

    assert keys == {("mnist", "example"), ("cartpole", "example")}


def test_record_submission_replay_writes_snake_replay(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(benchmark="snake-obstacles-8x8", input_indices=None)
    submission = SimpleNamespace(config=config)
    benchmark = SimpleNamespace(
        name="snake-obstacles-8x8",
        max_moves=123,
        load_submission=lambda value: SimpleNamespace(model="fake-model"),
        environment_options=lambda seed: {
            "size": 8,
            "blocked": ((1, 1),),
            "loop_on_full": True,
        },
    )
    captured = {}
    monkeypatch.setattr(workflow.Submission, "from_directory", lambda path: submission)
    monkeypatch.setattr(workflow, "create_benchmark", lambda name: benchmark)

    def fake_record(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return {"seed": 0, "frames": [{"snake": [[0, 0]], "food": [1, 1]}]}

    monkeypatch.setattr(workflow, "record_snake_game", fake_record)

    replay_path = workflow._record_submission_replay(
        tmp_path, Path("models/snake-obstacles-8x8/test-agent")
    )

    assert replay_path == (tmp_path / "docs" / "games" / "snake-obstacles-8x8--test-agent.json")
    assert replay_path.is_file()
    assert captured["max_moves"] == 123
    assert captured["blocked"] == ((1, 1),)
    assert captured["loop_on_full"] is True
    assert json.loads(replay_path.read_text())["seed"] == 0


def test_non_game_submission_does_not_record_replay(monkeypatch, tmp_path: Path) -> None:
    submission = SimpleNamespace(config=SimpleNamespace(benchmark="mnist"))
    monkeypatch.setattr(workflow.Submission, "from_directory", lambda path: submission)

    assert workflow._record_submission_replay(tmp_path, Path("models/mnist/test")) is None


def test_cleanup_failure_does_not_fail_acceptance(monkeypatch, tmp_path: Path) -> None:
    pr = PullRequest(7, "test", "a" * 40, "submission", "submit/test", True, checks_status="passed")
    result = _result("test", "mnist")
    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(workflow, "_sync_submission", lambda root: None)
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [pr])
    monkeypatch.setattr(
        workflow, "_validate_pr", lambda pr, root, base: ("a" * 40, Path("models/mnist/test"))
    )
    monkeypatch.setattr(
        workflow, "_score_candidate", lambda candidate, number, directory, root, data: result
    )
    monkeypatch.setattr(
        workflow,
        "_publish_result",
        lambda root, directory, scored, **kwargs: calls.append(("publish", None)),
    )
    monkeypatch.setattr(workflow, "_verify_promoted", lambda root: None)
    monkeypatch.setattr(workflow, "_is_ancestor", lambda ancestor, descendant, root: True)
    monkeypatch.setattr(
        workflow, "merge", lambda number, delete_branch, **kwargs: calls.append(("merge", number))
    )
    monkeypatch.setattr(workflow, "_verified_merge", lambda pr, root, expected: pr)
    monkeypatch.setattr(
        workflow,
        "delete_remote_branch",
        lambda number: (_ for _ in ()).throw(RuntimeError("cleanup unavailable")),
    )
    monkeypatch.setattr(workflow, "_is_merged_now", lambda pr: pr)
    comments: list[tuple[int, str]] = []
    monkeypatch.setattr(workflow, "comment", lambda number, body: comments.append((number, body)))

    def fake_git(args: list[str], cwd: Path) -> str:
        if args == ["branch", "--show-current"]:
            return "submission\n"
        if args == ["rev-parse", "origin/submission"]:
            return "b" * 40 + "\n"
        if args == ["rev-list", "--count", "origin/main..origin/submission"]:
            return "0\n"
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)

    assert workflow.run(tmp_path, None, Path("data")) == 0
    assert calls == [("merge", 7), ("publish", None)]
    assert comments == []


def test_publish_rebuilds_ledger_after_concurrent_push(monkeypatch, tmp_path: Path) -> None:
    result = _result("new-model", "mnist")
    local_keys = {("mnist", "existing")}
    remote_keys = set(local_keys)
    current_head = "a" * 40
    remote_head = "c" * 40
    push_count = 0

    def fake_git(args: list[str], cwd: Path) -> str:
        nonlocal current_head, push_count
        if args == ["rev-parse", "HEAD"]:
            return current_head + "\n"
        if args == ["rev-parse", "origin/submission"]:
            return remote_head + "\n"
        if args == ["push", "origin", "submission"]:
            push_count += 1
            if push_count == 1:
                remote_keys.add(("cartpole", "concurrent-model"))
                raise workflow.WorkflowError("non-fast-forward")
            remote_keys.update(local_keys)
            return ""
        if args == ["reset", "--hard", "origin/submission"]:
            current_head = remote_head
            local_keys.clear()
            local_keys.update(remote_keys)
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)
    monkeypatch.setattr(workflow, "_is_ancestor", lambda ancestor, descendant, root: True)
    monkeypatch.setattr(workflow, "_local_ledger_keys", lambda root: set(local_keys))
    monkeypatch.setattr(workflow, "_remote_ledger_keys", lambda root: set(remote_keys))
    monkeypatch.setattr(
        workflow,
        "upsert_result",
        lambda path, scored: local_keys.add(scoring.result_key(scored)),
    )
    monkeypatch.setattr(workflow, "_record_submission_replay", lambda root, directory: None)
    monkeypatch.setattr(workflow, "_render_site_atomically", lambda root, results: None)
    monkeypatch.setattr(workflow, "load_results", lambda path: [result])
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    workflow._publish_result(tmp_path, Path("models/mnist/new-model"), result)

    assert push_count == 2
    assert remote_keys == {
        ("mnist", "existing"),
        ("cartpole", "concurrent-model"),
        ("mnist", "new-model"),
    }


def test_remote_cleanup_leaves_non_submit_branches_alone(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_gh(args, *, check=True):
        calls.append(args)
        return json.dumps(
            {
                "headRefName": "feature/keep-this",
                "headRepository": {"name": "minimal-nn-competition"},
                "headRepositoryOwner": {"login": "wd7512"},
            }
        )

    monkeypatch.setattr(github, "_gh", fake_gh)

    github.delete_remote_branch(9)

    assert len(calls) == 1
    assert calls[0][:3] == ["pr", "view", "9"]


def test_recovery_revalidates_the_merged_commit(monkeypatch, tmp_path: Path, caplog) -> None:
    pr = PullRequest(8, "test", "a" * 40, "submission", "submit/test", True, checks_status="passed")
    merged = PullRequest(
        8,
        "test",
        "b" * 40,
        "submission",
        "submit/test",
        True,
        merged=True,
        merge_commit_sha="c" * 40,
    )
    result = _result("test", "mnist")
    published: list[str] = []
    comments: list[str] = []
    validations: list[bool] = []
    monkeypatch.setattr(workflow, "_sync_submission", lambda root: None)
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [pr])

    def validate(pr, root, base):
        validations.append(pr.merged)
        if pr.merged:
            raise workflow.WorkflowError("merged commit could not be validated")
        return "a" * 40, Path("models/mnist/test")

    monkeypatch.setattr(workflow, "_validate_pr", validate)
    monkeypatch.setattr(
        workflow, "_score_candidate", lambda candidate, number, directory, root, data: result
    )
    monkeypatch.setattr(workflow, "merge", lambda number, delete_branch, **kwargs: None)
    monkeypatch.setattr(
        workflow,
        "_verified_merge",
        lambda pr, root, expected: (_ for _ in ()).throw(
            workflow.WorkflowError("merge metadata unavailable")
        ),
    )
    monkeypatch.setattr(workflow, "_is_merged_now", lambda pr: merged)
    monkeypatch.setattr(
        workflow, "_publish_result", lambda root, directory, scored, **kwargs: published.append("x")
    )
    monkeypatch.setattr(workflow, "comment", lambda number, body: comments.append(body))
    monkeypatch.setattr(workflow, "_promote", lambda root: None)

    def fake_git(args: list[str], cwd: Path) -> str:
        if args == ["branch", "--show-current"]:
            return "submission\n"
        if args == ["rev-parse", "origin/submission"]:
            return "b" * 40 + "\n"
        return ""

    monkeypatch.setattr(workflow, "_git", fake_git)

    with caplog.at_level(logging.ERROR):
        assert workflow.run(tmp_path, None, Path("data")) == 0

    assert validations == [False, True]
    assert published == []
    assert comments == []
    assert "recovery still failed" in caplog.text
