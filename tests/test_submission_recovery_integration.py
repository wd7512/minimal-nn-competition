from __future__ import annotations

import json
import os
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


def _score_commit(work: Path, *, push: bool = True) -> None:
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
    _git(work, "commit", "-am", "Score submission scored-model")
    if push:
        _git(work, "push", "origin", "submission")
    _git(work, "fetch", "origin")


def _is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=cwd,
            check=False,
        ).returncode
        == 0
    )


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
        merged=True,
        merge_commit_sha=sha,
    )

    with pytest.raises(workflow.WorkflowError, match="two-parent merge commit"):
        workflow._validate_pr(pr, work, None)


def test_verified_merge_requires_the_validated_head(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    base = _rev(work, "submission")
    _git(work, "checkout", "-b", "feature")
    (work / "feature.txt").write_text("feature\n")
    _git(work, "add", "feature.txt")
    _git(work, "commit", "-m", "feature")
    _git(work, "checkout", "submission")
    _git(work, "merge", "--no-ff", "-m", "merge feature", "feature")
    head = _rev(work, "submission^2")
    merge_sha = _rev(work, "submission")
    merged = PullRequest(
        7,
        "merged",
        head,
        "submission",
        "feature",
        True,
        merged=True,
        merge_commit_sha=merge_sha,
    )
    monkeypatch.setattr(workflow, "get_pull_request", lambda number: merged)

    assert workflow._verified_merge(merged, work, head) is None

    stale = PullRequest(7, "stale", base, "submission", "feature", True)
    with pytest.raises(workflow.WorkflowError, match="does not match the validated commit"):
        workflow._verified_merge(stale, work, base)


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


def test_stale_pr_is_rejected_when_remote_advances(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _git(work, "checkout", "-b", "submit/stale")
    (work / "stale.txt").write_text("stale\n")
    _git(work, "add", "stale.txt")
    _git(work, "commit", "-m", "stale submission")
    head = _rev(work, "HEAD")
    _git(work, "push", "origin", "submit/stale")
    _git(work, "push", "origin", f"{head}:refs/pull/1/head")
    _git(work, "checkout", "submission")
    pr = PullRequest(1, "stale", head, "submission", "submit/stale", True, checks_status="passed")
    comments: list[str] = []

    def advance_then_list() -> list[PullRequest]:
        other = _clone(remote, tmp_path / "other")
        _git(other, "checkout", "submission")
        (other / "docs" / "leaderboard.json").write_text(
            '[{"benchmark": "mnist", "submission": "other"}]\n'
        )
        _git(other, "commit", "-am", "advance")
        _git(other, "push", "origin", "submission")
        return [pr]

    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", advance_then_list)
    monkeypatch.setattr(workflow, "_is_merged_now", lambda pr: pr)
    monkeypatch.setattr(workflow, "comment", lambda number, message: comments.append(message))
    monkeypatch.setattr(workflow, "_promote", lambda root: None)

    assert workflow._run(work, None, Path("data")) == 0
    assert comments
    assert "not based on the validation base" in comments[0]


def test_install_site_failure_restores_previous_tree(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    docs = work / "docs"
    (docs / "index.html").write_text("old index\n")
    (docs / ".nojekyll").write_text("")
    (docs / "submissions").mkdir()
    (docs / "submissions" / "keep.html").write_text("old page\n")
    (docs / "games").mkdir()
    (docs / "games" / "keep.json").write_text('{"frames": []}\n')
    _git(work, "add", "docs")
    _git(work, "commit", "-m", "existing site")

    real_replace = os.replace
    failed = False

    def fail_index_once(source, target) -> None:
        nonlocal failed
        if not failed and Path(source).name == "index.html":
            failed = True
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_index_once)
    monkeypatch.setattr(workflow, "_record_submission_replay", lambda root, directory: None)

    workflow._publish_result(work, Path("models/mnist/new-model"), _result("new-model"))

    assert failed
    assert (docs / "index.html").read_text() == "old index\n"
    assert (docs / "submissions" / "keep.html").read_text() == "old page\n"
    assert (docs / "games" / "keep.json").read_text() == '{"frames": []}\n'
    assert not list(docs.glob(".mnnc-*"))
    assert _git(work, "status", "--porcelain", "--untracked-files=all") == ""
    rows = json.loads(_git(work, "show", "origin/submission:docs/leaderboard.json"))
    assert ("mnist", "new-model") in {(row["benchmark"], row["submission"]) for row in rows}


def test_rerun_publishes_local_score_commit(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work, push=False)
    local = _rev(work, "submission")
    assert local != _rev(work, "origin/submission")

    other = _clone(remote, tmp_path / "other")
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "open_submission_prs", lambda: [])
    monkeypatch.setattr(workflow, "find_promotion_pr", lambda: None)
    monkeypatch.setattr(workflow, "create_promotion_pr", lambda: 11)

    def fake_merge(number: int) -> None:
        assert number == 11
        _git(other, "fetch", "origin")
        _git(other, "merge", "--no-ff", "-m", "promote", "origin/submission")
        _git(other, "push", "origin", "main")

    monkeypatch.setattr(workflow, "merge_promotion", fake_merge)

    assert workflow._run(work, None, Path("data")) == 0
    assert _is_ancestor(work, local, "origin/main")
    rows = json.loads(_git(work, "show", "origin/main:docs/leaderboard.json"))
    assert ("mnist", "scored-model") in {(row["benchmark"], row["submission"]) for row in rows}


def test_rerun_rejects_unpublished_manual_commit(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    (work / "notes.txt").write_text("manual\n")
    _git(work, "add", "notes.txt")
    _git(work, "commit", "-m", "manual change")
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)

    with pytest.raises(workflow.WorkflowError, match="not produced by the scoring workflow"):
        workflow._run(work, None, Path("data"))


def test_merged_recovery_uses_the_merge_parent_after_head_advances(
    tmp_path: Path, monkeypatch
) -> None:
    _remote, work = _repo(tmp_path)
    _git(work, "checkout", "-b", "feature")
    submission_dir = work / "models" / "mnist" / "merged"
    submission_dir.mkdir(parents=True)
    (submission_dir / "config.json").write_text("{}\n")
    _git(work, "add", "models")
    _git(work, "commit", "-m", "submission")
    _git(work, "push", "origin", "feature")
    head = _rev(work, "HEAD")
    _git(work, "checkout", "submission")
    _git(work, "merge", "--no-ff", "-m", "merge feature", "feature")
    merge_sha = _rev(work, "submission")
    _git(work, "push", "origin", "submission")

    _git(work, "checkout", "feature")
    (work / "advanced.txt").write_text("advanced\n")
    _git(work, "add", "advanced.txt")
    _git(work, "commit", "-m", "advance")
    advanced = _rev(work, "HEAD")
    _git(work, "push", "origin", "feature")
    _git(work, "checkout", "submission")
    monkeypatch.setattr(workflow, "validate_diff", lambda diff, worktree: [])

    pr = PullRequest(
        9,
        "advanced",
        advanced,
        "submission",
        "feature",
        True,
        merged=True,
        merge_commit_sha=merge_sha,
    )
    candidate, directory = workflow._validate_pr(pr, work, None)

    assert candidate == head
    assert candidate != pr.head_sha
    assert directory == Path("models/mnist/merged")
