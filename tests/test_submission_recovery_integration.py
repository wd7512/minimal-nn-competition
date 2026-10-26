from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimal_nn_competition.benchmarks.snake.replay import replay_stem
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions import workflow
from minimal_nn_competition.submissions.github import PullRequest


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rev(cwd: Path, ref: str) -> str:
    return _git(cwd, "rev-parse", ref)


def _clone(remote: Path, target: Path) -> Path:
    subprocess.run(["git", "clone", str(remote), str(target)], check=True, capture_output=True)
    _git(target, "config", "user.email", "test@example.com")
    _git(target, "config", "user.name", "Test")
    return target


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote and a working clone with main and submission branches."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    work = _clone(remote, tmp_path / "work")
    _git(work, "checkout", "-b", "submission")
    (work / "docs").mkdir()
    (work / "docs" / "leaderboard.json").write_text("[]\n")
    _git(work, "add", "docs/leaderboard.json")
    _git(work, "commit", "-m", "initial")
    _git(work, "push", "-u", "origin", "submission")
    _git(work, "checkout", "-b", "main")
    _git(work, "push", "-u", "origin", "main")
    _git(work, "checkout", "submission")
    _git(work, "fetch", "origin")
    subprocess.run(
        ["git", "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
    )
    return remote, work


def _score_commit(work: Path) -> None:
    submission = _result("scored-model")
    rows = [
        {
            "submission": submission.submission,
            "benchmark": submission.benchmark,
            "score": submission.score,
            "passed": submission.passed,
            "parameter_count": submission.parameter_count,
            "details": submission.details,
            "name": submission.name,
            "author": submission.author,
        }
    ]
    (work / "docs" / "leaderboard.json").write_text(json.dumps(rows) + "\n")
    _git(work, "commit", "-am", "score")
    _git(work, "push", "origin", "submission")
    _git(work, "fetch", "origin")


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


def test_sync_submission_recovers_after_promotion_merge(tmp_path: Path) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work)
    other = _clone(remote, tmp_path / "other")
    _git(other, "merge", "--no-ff", "-m", "promote", "origin/submission")
    _git(other, "push", "origin", "main")

    workflow._sync_submission(work)

    assert _rev(work, "submission") == _rev(work, "origin/main")
    assert _rev(work, "origin/submission") == _rev(work, "origin/main")


def test_promote_resumes_existing_promotion_pr(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work)
    other = _clone(remote, tmp_path / "other")
    created: list[int] = []

    monkeypatch.setattr(workflow, "find_promotion_pr", lambda: 42)
    monkeypatch.setattr(workflow, "create_promotion_pr", lambda: created.append(1) or 1)

    def fake_merge(number: int) -> None:
        assert number == 42
        _git(other, "fetch", "origin")
        _git(other, "merge", "--no-ff", "-m", "promote", "origin/submission")
        _git(other, "push", "origin", "main")

    monkeypatch.setattr(workflow, "merge_promotion", fake_merge)

    workflow._promote(work)

    assert created == []
    assert _rev(work, "origin/submission") == _rev(work, "origin/main")


def test_promote_creates_pr_when_none_exists(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work)
    other = _clone(remote, tmp_path / "other")
    created: list[int] = []

    monkeypatch.setattr(workflow, "find_promotion_pr", lambda: None)
    monkeypatch.setattr(workflow, "create_promotion_pr", lambda: created.append(7) or 7)

    def fake_merge(number: int) -> None:
        assert number == 7
        _git(other, "fetch", "origin")
        _git(other, "merge", "--no-ff", "-m", "promote", "origin/submission")
        _git(other, "push", "origin", "main")

    monkeypatch.setattr(workflow, "merge_promotion", fake_merge)

    workflow._promote(work)

    assert created == [7]
    assert _rev(work, "origin/submission") == _rev(work, "origin/main")


def test_promote_is_idempotent_after_partial_promotion(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work)
    other = _clone(remote, tmp_path / "other")
    _git(other, "merge", "--no-ff", "-m", "promote", "origin/submission")
    _git(other, "push", "origin", "main")
    workflow._sync_submission(work)

    monkeypatch.setattr(
        workflow, "find_promotion_pr", lambda: pytest.fail("must not look for a new promotion")
    )
    monkeypatch.setattr(
        workflow, "create_promotion_pr", lambda: pytest.fail("must not create a new promotion")
    )
    monkeypatch.setattr(
        workflow, "merge_promotion", lambda number: pytest.fail("must not merge again")
    )

    workflow._promote(work)

    assert _rev(work, "origin/submission") == _rev(work, "origin/main")


def test_stage_generated_ignores_unrelated_docs_files(tmp_path: Path) -> None:
    _remote, work = _repo(tmp_path)
    (work / "docs" / "index.html").write_text("new index\n")
    (work / "docs" / "notes.txt").write_text("stray\n")
    (work / "docs" / "games").mkdir()
    (work / "docs" / "games" / "stray.json").write_text("{}\n")

    workflow._stage_generated(work, ("docs/index.html",))

    assert _git(work, "diff", "--cached", "--name-only").splitlines() == ["docs/index.html"]


def test_publish_still_records_score_when_presentation_fails(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    result = _result("new-model")

    def fail_replay(root: Path, directory: Path) -> None:
        raise RuntimeError("replay unavailable")

    def fail_render(results, output_dir, replays=None) -> None:
        raise ValueError("render unavailable")

    monkeypatch.setattr(workflow, "_record_submission_replay", fail_replay)
    monkeypatch.setattr(workflow, "render_site", fail_render)

    workflow._publish_result(work, Path("models/mnist/new-model"), result, pr_number=99)

    assert _git(work, "log", "-1", "--pretty=%s") == "Score submission PR #99 (new-model)"
    committed = _git(work, "show", "--name-only", "--pretty=format:", "HEAD").splitlines()
    assert committed == ["docs/leaderboard.json"]
    payload = json.loads(_git(work, "show", "origin/submission:docs/leaderboard.json"))
    assert {(row["benchmark"], row["submission"]) for row in payload} == {("mnist", "new-model")}


def test_publish_does_not_swallow_programming_errors(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)

    def bug(root: Path, directory: Path) -> None:
        raise KeyError("programming error")

    monkeypatch.setattr(workflow, "_record_submission_replay", bug)

    with pytest.raises(KeyError):
        workflow._publish_result(work, Path("models/mnist/broken"), _result("broken"))


def test_concurrent_runs_cannot_share_a_checkout(tmp_path: Path) -> None:
    with workflow._exclusive_run_lock(tmp_path):  # noqa: SIM117 - outer lock must stay held
        with pytest.raises(workflow.WorkflowError, match="already active"):
            with workflow._exclusive_run_lock(tmp_path):
                pass


def test_merged_pr_recovery_rejects_single_parent_commit(tmp_path: Path) -> None:
    _remote, work = _repo(tmp_path)
    sha = _rev(work, "HEAD")
    pr = PullRequest(
        5,
        "squashed",
        sha,
        "submission",
        "submit/model",
        True,
        state="MERGED",
        merged=True,
        merge_commit_sha=sha,
    )

    with pytest.raises(workflow.WorkflowError, match="two-parent merge commit"):
        workflow._validate_pr(pr, work, None)


def test_load_replays_ignores_unrelated_files(tmp_path: Path) -> None:
    _remote, work = _repo(tmp_path)
    games = work / "docs" / "games"
    games.mkdir()
    result = SimpleNamespace(benchmark="snake-simple-8x8", submission="agent")
    stem = replay_stem(result.benchmark, result.submission)
    (games / f"{stem}.json").write_text('{"frames": []}\n')
    (games / "stray.json").write_text("not json\n")

    replays = workflow._load_replays(work, [result])

    assert set(replays) == {stem}
