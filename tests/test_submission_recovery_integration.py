from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimal_nn_competition.benchmarks.common.snake.replay import replay_stem
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


def test_install_site_preserves_unrendered_files_in_generated_directories(
    tmp_path: Path,
) -> None:
    _remote, work = _repo(tmp_path)
    games = work / "docs" / "games"
    submissions = work / "docs" / "submissions"
    games.mkdir()
    submissions.mkdir()
    (games / "notes.json").write_text('{"local": true}\n')
    (submissions / "notes.txt").write_text("keep me\n")
    rendered = tmp_path / "rendered"
    (rendered / "games").mkdir(parents=True)
    (rendered / "submissions").mkdir()
    for name in workflow._SITE_FILES:
        (rendered / name).write_text("rendered\n")
    (rendered / "games" / "known.json").write_text('{"rendered": true}\n')

    workflow._install_site(rendered, work / "docs")

    assert (games / "notes.json").read_text() == '{"local": true}\n'
    assert (submissions / "notes.txt").read_text() == "keep me\n"
    assert (games / "known.json").is_file()


def test_render_failure_stages_only_the_ledger(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    result = _result("new-model")
    (work / "docs" / "games").mkdir(parents=True)
    (work / "docs" / "games" / "notes.json").write_text("{}\n")

    def fail_replay(root: Path, directory: Path, scored_sha: str) -> None:
        raise RuntimeError("replay unavailable")

    def fail_render(results, output_dir, replays=None) -> None:
        raise ValueError("render unavailable")

    monkeypatch.setattr(workflow, "_record_submission_replay", fail_replay)
    monkeypatch.setattr(workflow, "render_site", fail_render)

    scored_sha = _rev(work, "submission")
    workflow._publish_result(
        work, Path("models/mnist/new-model"), result, scored_sha=scored_sha, pr_number=99
    )

    assert _git(work, "log", "-1", "--pretty=%s") == "Score submission PR #99 (new-model)"
    committed = _git(work, "show", "--name-only", "--pretty=format:", "HEAD").splitlines()
    assert committed == ["docs/leaderboard.json"]
    assert "docs/games/notes.json" not in committed
    assert (work / "docs" / "games" / "notes.json").is_file()
    payload = json.loads(_git(work, "show", "origin/submission:docs/leaderboard.json"))
    assert {(row["benchmark"], row["submission"]) for row in payload} == {("mnist", "new-model")}


def test_replay_write_failure_preserves_previous_replay_on_render_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _remote, work = _repo(tmp_path)
    result = _result("agent", "snake-simple-8x8")
    replay_path = (
        work / "docs" / "games" / f"{replay_stem(result.benchmark, result.submission)}.json"
    )
    replay_path.parent.mkdir(parents=True)
    previous_replay = '{"frames": [{"snake": [[0, 0]], "food": [1, 1]}]}\n'
    replay_path.write_text(previous_replay)

    def fail_replay(root: Path, directory: Path, scored_sha: str) -> None:
        replay_path.write_text('{"frames": [')
        raise OSError("disk full")

    def fail_render(results, output_dir, replays=None) -> None:
        raise ValueError("render unavailable")

    monkeypatch.setattr(workflow, "_record_submission_replay", fail_replay)
    monkeypatch.setattr(workflow, "render_site", fail_render)

    workflow._publish_result(
        work, Path("models/snake-simple-8x8/agent"), result, scored_sha=_rev(work, "submission")
    )

    committed_replay = _git(work, "show", f"HEAD:docs/games/{replay_path.name}")
    assert committed_replay == previous_replay.strip()
    assert replay_path.read_text() == previous_replay


def test_commit_failure_restores_artifacts_and_allows_retry(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    result = _result("retry-model")
    original_ledger = (work / "docs" / "leaderboard.json").read_bytes()
    original_head = _rev(work, "HEAD")
    fail_commit = True
    original_git = workflow._git

    def git(args: list[str], cwd: Path) -> str:
        nonlocal fail_commit
        if args[:1] == ["commit"] and fail_commit:
            fail_commit = False
            raise workflow.WorkflowError("simulated commit failure")
        return original_git(args, cwd)

    def record_replay(root: Path, directory: Path, scored_sha: str) -> Path:
        path = root / "docs" / "games" / f"{replay_stem(result.benchmark, result.submission)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"frames": []}\n')
        return path

    def render(root: Path, results) -> None:
        (root / "docs" / "index.html").write_text("rendered\n")
        pages = root / "docs" / "submissions"
        pages.mkdir(parents=True, exist_ok=True)
        (pages / "retry-model.html").write_text("rendered\n")

    monkeypatch.setattr(workflow, "_git", git)
    monkeypatch.setattr(workflow, "_record_submission_replay", record_replay)
    monkeypatch.setattr(workflow, "_render_site_atomically", render)

    with pytest.raises(workflow.WorkflowError, match="simulated commit failure"):
        workflow._publish_result(
            work, Path("models/mnist/retry-model"), result, scored_sha=original_head
        )

    assert _rev(work, "HEAD") == original_head
    assert (work / "docs" / "leaderboard.json").read_bytes() == original_ledger
    assert _git(work, "status", "--porcelain", "--untracked-files=all") == ""

    workflow._publish_result(
        work, Path("models/mnist/retry-model"), result, scored_sha=original_head
    )

    assert _git(work, "log", "-1", "--pretty=%s") == "Score submission retry-model"
    assert "retry-model" in _git(work, "show", "HEAD:docs/leaderboard.json")


def test_failed_publication_rollback_retains_backup(tmp_path: Path, monkeypatch, caplog) -> None:
    _remote, work = _repo(tmp_path)
    original_head = _rev(work, "HEAD")
    original_ledger = (work / "docs" / "leaderboard.json").read_bytes()
    original_git = workflow._git
    backups: list[Path] = []

    def fail_commit(args: list[str], cwd: Path) -> str:
        if args[:1] == ["commit"]:
            raise workflow.WorkflowError("simulated commit failure")
        return original_git(args, cwd)

    def fail_restore(root: Path, backup: Path, existing: tuple[str, ...]) -> None:
        backups.append(backup)
        raise OSError("simulated restore failure")

    monkeypatch.setattr(workflow, "_git", fail_commit)
    monkeypatch.setattr(workflow, "_restore_generated", fail_restore)

    with pytest.raises(workflow.WorkflowError, match="recovery backup retained"):
        workflow._publish_result(
            work,
            Path("models/mnist/backup-model"),
            _result("backup-model"),
            scored_sha=original_head,
        )

    assert len(backups) == 1
    assert (backups[0] / "docs" / "leaderboard.json").read_bytes() == original_ledger
    assert f"recovery backup retained at {backups[0]}" in caplog.text


def test_push_failure_keeps_score_commit_for_recovery(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    result = _result("push-retry")
    original_head = _rev(work, "HEAD")
    original_git = workflow._git

    def fail_push(args: list[str], cwd: Path) -> str:
        if args == ["push", "origin", "submission"]:
            raise workflow.WorkflowError("simulated push failure")
        return original_git(args, cwd)

    monkeypatch.setattr(workflow, "_git", fail_push)
    monkeypatch.setattr(
        workflow, "_record_submission_replay", lambda root, directory, scored_sha: None
    )
    monkeypatch.setattr(workflow, "_render_site_atomically", lambda root, results: None)

    with pytest.raises(workflow.WorkflowError, match="simulated push failure"):
        workflow._publish_result(
            work, Path("models/mnist/push-retry"), result, scored_sha=original_head
        )

    assert _rev(work, "HEAD") != original_head
    assert _rev(work, "origin/submission") == original_head
    assert "push-retry" in _git(work, "show", "HEAD:docs/leaderboard.json")
    assert _git(work, "status", "--porcelain", "--untracked-files=all") == ""


def test_publish_does_not_swallow_programming_errors(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)

    def bug(root: Path, directory: Path, scored_sha: str) -> None:
        raise KeyError("programming error")

    monkeypatch.setattr(workflow, "_record_submission_replay", bug)

    with pytest.raises(KeyError):
        workflow._publish_result(
            work, Path("models/mnist/broken"), _result("broken"), scored_sha="a" * 40
        )


def test_publish_does_not_swallow_rendering_programming_errors(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)

    def bug(results, output_dir, replays=None) -> None:
        raise KeyError("programming error")

    monkeypatch.setattr(workflow, "render_site", bug)

    with pytest.raises(KeyError):
        workflow._publish_result(
            work, Path("models/mnist/broken"), _result("broken"), scored_sha="a" * 40
        )


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


def test_accept_pr_rejects_unmerged_pr(tmp_path: Path) -> None:
    pr = PullRequest(1, "open", "a" * 40, "submission", "submit/open", True)

    with pytest.raises(workflow.WorkflowError, match="not merged"):
        workflow._accept_pr(pr, tmp_path, Path("data"))


def test_replay_uses_the_scored_checkpoint(tmp_path: Path, monkeypatch) -> None:
    _remote, work = _repo(tmp_path)
    directory = Path("models/snake-simple-8x8/agent")
    checkpoint = work / directory / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("old-model\n")
    _git(work, "add", "models")
    _git(work, "commit", "-m", "old model")
    scored_sha = _rev(work, "HEAD")

    checkpoint.write_text("new-model\n")
    _git(work, "commit", "-am", "new model")
    assert _rev(work, "HEAD") != scored_sha

    monkeypatch.setattr(
        workflow.Submission,
        "from_directory",
        classmethod(
            lambda cls, path: SimpleNamespace(
                config=SimpleNamespace(benchmark="snake-simple-8x8", input_indices=None),
                checkpoint=path / "model.pt",
            )
        ),
    )

    class FakeBenchmark:
        max_moves = 1

        def load_submission(self, submission):
            return SimpleNamespace(model=submission.checkpoint.read_text().strip())

        def environment_options(self, seed):
            return {}

    monkeypatch.setattr(workflow, "create_benchmark", lambda name: FakeBenchmark())
    monkeypatch.setattr(
        workflow,
        "record_snake_game",
        lambda model, **kwargs: {"frames": [], "model": model},
    )

    replay = workflow._record_submission_replay(work, directory, scored_sha)

    assert replay is not None
    assert (work / directory / "model.pt").read_text().strip() == "new-model"
    assert json.loads(replay.read_text())["model"] == "old-model"


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
    monkeypatch.setattr(
        workflow, "_record_submission_replay", lambda root, directory, scored_sha: None
    )

    workflow._publish_result(
        work,
        Path("models/mnist/new-model"),
        _result("new-model"),
        scored_sha=_rev(work, "submission"),
    )

    assert failed
    assert (docs / "index.html").read_text() == "old index\n"
    assert (docs / "submissions" / "keep.html").read_text() == "old page\n"
    assert (docs / "games" / "keep.json").read_text() == '{"frames": []}\n'
    assert not list(docs.glob(".mnnc-*"))
    assert _git(work, "status", "--porcelain", "--untracked-files=all") == ""
    rows = json.loads(_git(work, "show", "origin/submission:docs/leaderboard.json"))
    assert ("mnist", "new-model") in {(row["benchmark"], row["submission"]) for row in rows}


def test_install_site_retains_backup_when_restore_fails(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    output = tmp_path / "docs"
    output.mkdir()
    old_ledger = output / "leaderboard.json"
    old_ledger.write_text("old ledger\n")
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    for name in workflow._SITE_FILES:
        (rendered / name).write_text("new output\n")
    for name in workflow._SITE_DIRS:
        (rendered / name).mkdir()

    real_replace = os.replace
    backups: list[Path] = []

    def fail_install_and_restore(source, target) -> None:
        source = Path(source)
        target = Path(target)
        if source.parent.name.startswith(".mnnc-stage-") and target == old_ledger:
            raise OSError("simulated install failure")
        if source.parent.name.startswith(".mnnc-backup-"):
            backups.append(source.parent)
            raise OSError("simulated restore failure")
        real_replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_install_and_restore)

    with pytest.raises(OSError, match="simulated restore failure"):
        workflow._install_site(rendered, output)

    assert backups
    assert (backups[0] / "leaderboard.json").read_text() == "old ledger\n"
    assert f"backup retained at {backups[0]}" in caplog.text


def test_rerun_publishes_local_score_commit(tmp_path: Path, monkeypatch) -> None:
    remote, work = _repo(tmp_path)
    _score_commit(work, push=False)
    local = _rev(work, "submission")
    assert local != _rev(work, "origin/submission")

    other = _clone(remote, tmp_path / "other")
    monkeypatch.setattr(workflow, "check_authentication", lambda: None)
    monkeypatch.setattr(workflow, "merged_submission_prs", lambda: [])
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
