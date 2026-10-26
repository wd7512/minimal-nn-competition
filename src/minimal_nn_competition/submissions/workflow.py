"""Sequential submission acceptance workflow."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from minimal_nn_competition.benchmarks.control.names import RL_BENCHMARKS
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import selected_input_indices
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.benchmarks.snake.replay import (
    REPLAY_SEED as SNAKE_REPLAY_SEED,
)
from minimal_nn_competition.benchmarks.snake.replay import (
    load_replay,
    replay_stem,
    save_replay,
)
from minimal_nn_competition.benchmarks.snake.replay import (
    record_game as record_snake_game,
)
from minimal_nn_competition.leaderboard import load_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.submissions.github import (
    GitHubError,
    PullRequest,
    check_authentication,
    comment,
    create_promotion_pr,
    delete_remote_branch,
    find_promotion_pr,
    get_pull_request,
    merge,
    merge_promotion,
    open_submission_prs,
)
from minimal_nn_competition.submissions.scoring import result_key, score_submission, upsert_result
from minimal_nn_competition.submissions.validation import submission_directory, validate_diff

logger = logging.getLogger(__name__)
# Only expected operational failures are recoverable. Programming errors such as
# KeyError or TypeError deliberately escape so they stay diagnosable.
_RECOVERY_ERRORS = (GitHubError, OSError, RuntimeError, ValueError)
_GENERATED_PATHS = (
    "docs/leaderboard.json",
    "docs/index.html",
    "docs/.nojekyll",
    "docs/submissions",
    "docs/games",
)
_SITE_FILES = ("leaderboard.json", "index.html", ".nojekyll")
_SITE_DIRS = ("submissions", "games")


class WorkflowError(RuntimeError):
    """Raised when the acceptance workflow cannot safely continue."""


@contextmanager
def _exclusive_run_lock(root: Path):
    """Refuse concurrent runs that would share this checkout."""
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"mnnc-submission-{digest}.lock"
    with open(lock_path, "a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise WorkflowError(
                "another submission validation run is already active for this repository"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"git {' '.join(args)} failed: {(exc.stderr or '').strip()}") from exc


def _is_ancestor(ancestor: str, descendant: str, root: Path) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        detail = (result.stderr or result.stdout).strip()
        raise WorkflowError(f"could not compare {ancestor} and {descendant}: {detail}")
    return result.returncode == 0


def _ledger_keys_from_payload(payload: str, source: str) -> set[tuple[str, str]]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"could not parse leaderboard from {source}: {exc}") from exc
    if not isinstance(rows, list):
        raise WorkflowError(f"leaderboard from {source} must be a JSON list")
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("benchmark") or not row.get("submission"):
            raise WorkflowError(f"leaderboard from {source} contains an invalid row")
        keys.add((str(row["benchmark"]), str(row["submission"])))
    return keys


def _local_ledger_keys(root: Path) -> set[tuple[str, str]]:
    path = root / "docs" / "leaderboard.json"
    if not path.exists():
        return set()
    return {result_key(result) for result in load_results(path)}


def _remote_ledger_keys(root: Path, ref: str = "origin/submission") -> set[tuple[str, str]]:
    payload = _git(["show", f"{ref}:docs/leaderboard.json"], root)
    return _ledger_keys_from_payload(payload, ref)


def _is_generated_path(path: str) -> bool:
    return path in _GENERATED_PATHS or path.startswith(("docs/submissions/", "docs/games/"))


def _unpublished_score_commits(root: Path) -> list[str]:
    """Return local commits that are neither on origin/submission nor already on main."""
    return _git(["rev-list", "origin/submission..submission", "--not", "origin/main"], root).split()


def _publish_local_score_commits(root: Path) -> None:
    """Push workflow-generated score commits left behind by a failed push."""
    commits = _unpublished_score_commits(root)
    if not commits:
        return
    for sha in commits:
        subject = _git(["log", "-1", "--pretty=%s", sha], root)
        paths = _git(["diff-tree", "--no-commit-id", "--name-only", "-r", sha], root).splitlines()
        if not subject.startswith("Score submission ") or not paths:
            raise WorkflowError(
                "local submission has unpublished commits that were not produced by the scoring "
                f"workflow ({subject or sha}). Push or reconcile them manually before rerunning."
            )
        if not all(_is_generated_path(path) for path in paths):
            raise WorkflowError(
                "local submission has an unpublished score commit that touches files outside "
                f"generated artifacts ({sha}). Reconcile it manually before rerunning."
            )
    try:
        _git(["push", "origin", "submission"], root)
    except WorkflowError as exc:
        raise WorkflowError(
            "local score commits could not be pushed to origin/submission; the result is not "
            "published yet. Fix the push failure and rerun."
        ) from exc
    _git(["fetch", "origin", "submission"], root)
    logger.info("Published %d local score commit(s)", len(commits))


def _sync_submission(root: Path) -> None:
    """Fetch and reconcile branches without rewriting or force-pushing history."""
    _git(["fetch", "origin", "main", "submission"], root)

    main_ahead = _is_ancestor("origin/submission", "origin/main", root)
    submission_ahead = _is_ancestor("origin/main", "origin/submission", root)
    if main_ahead and not submission_ahead:
        # A promotion merged but the final synchronisation of `submission` did not
        # finish. Fast-forward only; never reset away local unpushed work.
        if _is_ancestor("submission", "origin/main", root):
            _git(["merge", "--ff-only", "origin/main"], root)
            _git(["push", "origin", "submission"], root)
            _git(["fetch", "origin", "submission"], root)
            logger.info("Fast-forwarded submission to the promoted main tip")
        elif _is_ancestor("origin/main", "submission", root):
            _publish_local_score_commits(root)
        else:
            raise WorkflowError(
                "local submission diverged from the promoted main tip. Reconcile them "
                "explicitly; the acceptance workflow will not reset, rebase, or force-push submission."
            )
    elif submission_ahead:
        if _is_ancestor("origin/submission", "submission", root):
            _publish_local_score_commits(root)
        elif _is_ancestor("submission", "origin/submission", root):
            _git(["merge", "--ff-only", "origin/submission"], root)
        else:
            raise WorkflowError(
                "local submission and origin/submission have diverged. Reconcile them explicitly; "
                "the acceptance workflow will not reset, rebase, or force-push submission."
            )
    else:
        raise WorkflowError(
            "origin/submission and origin/main have diverged. Promote or reconcile the branches "
            "using the normal PR flow before accepting submissions; the workflow will not merge "
            "main into submission or rewrite either branch during acceptance."
        )

    # A local commit left behind by a previous run must not be allowed to erase rows
    # present at the current remote tip.
    local_keys = _local_ledger_keys(root)
    remote_keys = _remote_ledger_keys(root)
    if not remote_keys.issubset(local_keys):
        missing = sorted(remote_keys - local_keys)
        raise WorkflowError(
            "local leaderboard is missing entries from origin/submission; refusing to push "
            f"a possibly stale ledger. Missing keys: {missing[:8]}"
        )


def _current_validation_base(root: Path) -> str:
    return _git(["rev-parse", "origin/submission"], root).strip()


def _validate_pr(pr: PullRequest, root: Path, validation_base: str | None = None) -> Path:
    if pr.base_ref != "submission":
        raise WorkflowError("PR does not target submission")
    if not pr.merged and not pr.checks_passed:
        raise WorkflowError(f"required GitHub checks have not passed ({pr.checks_summary})")

    if pr.merged:
        if not pr.merge_commit_sha:
            raise WorkflowError(
                "PR is merged but GitHub did not return its merge commit; rerun once metadata is available"
            )
        # Recovery diffs the merge commit against its first parent. That is only the
        # full PR change set for true merge commits; squash and rebase merges need a
        # strategy-aware diff, so they are rejected rather than scored incorrectly.
        parents = _git(["rev-list", "--parents", "-n", "1", pr.merge_commit_sha], root).split()
        if len(parents) < 3:
            raise WorkflowError(
                "merged PR does not have a two-parent merge commit. Recovery supports only "
                "merge commits (the workflow's merge strategy); reconcile other merges manually."
            )
        diff = _git(
            ["diff", "--name-status", f"{pr.merge_commit_sha}^1", pr.merge_commit_sha], root
        )
    else:
        # A three-dot diff is only safe when the current integration tip is the PR base
        # (or its ancestor). Otherwise unrelated files can leak into the candidate diff.
        base = validation_base or "origin/submission"
        if not _is_ancestor(base, pr.head_sha, root):
            raise WorkflowError(
                f"PR head is not based on the validation base {base}. Rebase the submission "
                "branch onto the current origin/submission tip, then rerun validation."
            )
        diff = _git(["diff", "--name-status", f"{base}...{pr.head_sha}"], root)

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


def _record_submission_replay(root: Path, directory: Path) -> Path | None:
    """Write the deterministic detail-page replay for an accepted game submission."""
    submission = Submission.from_directory(root / directory)
    benchmark_name = submission.config.benchmark
    if benchmark_name not in SNAKE_BENCHMARKS + RL_BENCHMARKS:
        return None

    benchmark = create_benchmark(benchmark_name)
    loaded = benchmark.load_submission(submission)
    name = directory.name
    replay_dir = root / "docs" / "games"
    stem = replay_stem(benchmark_name, name)

    if benchmark_name in SNAKE_BENCHMARKS:
        payload = record_snake_game(
            loaded.model,
            seed=SNAKE_REPLAY_SEED,
            max_moves=benchmark.max_moves,
            input_indices=selected_input_indices(submission.config, benchmark_name),
            **benchmark.environment_options(SNAKE_REPLAY_SEED),
        )
    else:
        from minimal_nn_competition.benchmarks.control.replay import record_game

        payload = record_game(benchmark_name, loaded.model, steps=benchmark.max_steps)

    return save_replay(replay_dir / f"{stem}.json", payload)


def _expected_replay_stems(results) -> set[str]:
    return {
        replay_stem(result.benchmark, result.submission)
        for result in results
        if result.benchmark in SNAKE_BENCHMARKS + RL_BENCHMARKS
    }


def _load_replays(root: Path, results=None) -> dict:
    """Load replays for the known leaderboard rows only, ignoring stray files."""
    replay_dir = root / "docs" / "games"
    if not replay_dir.is_dir():
        return {}
    if results is None:
        paths = sorted(replay_dir.glob("*.json"))
    else:
        paths = sorted(replay_dir / f"{stem}.json" for stem in _expected_replay_stems(results))
    replays = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            replays[path.stem] = load_replay(path)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable replay %s: %s", path, exc)
    return replays


def _install_site(rendered: Path, output_dir: Path) -> None:
    """Swap rendered artifacts into place, restoring the previous tree on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mnnc-stage-", dir=output_dir))
    backup = Path(tempfile.mkdtemp(prefix=".mnnc-backup-", dir=output_dir))
    touched: list[str] = []
    try:
        for name in _SITE_FILES:
            shutil.copyfile(rendered / name, staging / name)
        for name in _SITE_DIRS:
            source = rendered / name
            if source.is_dir():
                shutil.copytree(source, staging / name)
            else:
                (staging / name).mkdir()
        for name in _SITE_FILES + _SITE_DIRS:
            target = output_dir / name
            if target.exists() or target.is_symlink():
                os.replace(target, backup / name)
            touched.append(name)
            os.replace(staging / name, target)
    except BaseException:
        for name in reversed(touched):
            target = output_dir / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists() or target.is_symlink():
                target.unlink()
            saved = backup / name
            if saved.exists() or saved.is_symlink():
                os.replace(saved, target)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _render_site_atomically(root: Path, results) -> None:
    """Render to a scratch directory, then swap outputs into place with rollback."""
    output_dir = root / "docs"
    with tempfile.TemporaryDirectory(prefix="mnnc-render-") as temp:
        rendered = Path(temp) / "docs"
        render_site(results, rendered, _load_replays(root, results))
        _install_site(rendered, output_dir)


def _stage_generated(root: Path, paths: tuple[str, ...]) -> None:
    """Stage only the given generated artifacts, never arbitrary docs files."""
    if not paths:
        return
    status = _git(
        ["status", "--porcelain", "--untracked-files=all", "--", *paths],
        root,
    )
    staged = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        staged.append(path)
    if staged:
        _git(["add", "-A", "--", *sorted(set(staged))], root)


def _publish_result(root: Path, directory: Path, result, *, pr_number: int | None = None) -> None:
    """Upsert and publish artifacts, rebasing the generated commit on concurrent writes."""
    ledger_path = root / "docs" / "leaderboard.json"

    for attempt in range(3):
        base_sha = _git(["rev-parse", "HEAD"], root).strip()
        old_keys = _local_ledger_keys(root)
        remote_keys = _remote_ledger_keys(root)
        if not remote_keys.issubset(old_keys):
            raise WorkflowError(
                "refusing leaderboard update: local ledger is missing entries from "
                "origin/submission"
            )

        upsert_result(ledger_path, result)
        new_keys = _local_ledger_keys(root)
        if not old_keys.issubset(new_keys):
            raise WorkflowError("leaderboard upsert removed an existing entry")
        if result_key(result) not in new_keys:
            raise WorkflowError("leaderboard upsert did not record the scored submission")

        try:
            _record_submission_replay(root, directory)
        except _RECOVERY_ERRORS as replay_error:
            # Presentation artifacts must not strand an already-scored submission.
            logger.error(
                "Could not record replay for %s: %s. The score will still be published; "
                "rerun to repair the replay artifact.",
                result.submission,
                replay_error,
            )

        results = load_results(ledger_path)
        rendered = True
        try:
            _render_site_atomically(root, results)
        except Exception as render_error:
            rendered = False
            logger.exception("Could not render site for %s: %s", result.submission, render_error)

        if rendered:
            generated = _GENERATED_PATHS
        else:
            generated = ("docs/leaderboard.json",)
            for stem in sorted(_expected_replay_stems(results)):
                path = root / "docs" / "games" / f"{stem}.json"
                if path.is_file():
                    generated += (f"docs/games/{stem}.json",)
        _stage_generated(root, generated)

        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False)
        if staged.returncode == 1:
            message = f"Score submission {result.submission}"
            if pr_number is not None:
                message = f"Score submission PR #{pr_number} ({result.submission})"
            _git(["commit", "-m", message], root)
        elif staged.returncode != 0:
            raise WorkflowError("could not inspect staged leaderboard changes")

        try:
            _git(["push", "origin", "submission"], root)
        except WorkflowError as push_error:
            _git(["fetch", "origin", "submission"], root)
            remote_tip = _git(["rev-parse", "origin/submission"], root).strip()
            if remote_tip == base_sha:
                # The push failed without a competing write (e.g. permission failure).
                # Keep the local score commit so an explicit recovery run can retry it.
                raise push_error
            if not _is_ancestor(base_sha, remote_tip, root):
                raise WorkflowError(
                    "push failed and origin/submission did not advance from the publish base; "
                    "refusing to reset potentially unrelated local commits"
                ) from push_error
            # Another run advanced the integration branch after our ledger read. This
            # attempt's commit is workflow-generated and is the only commit after base_sha.
            # Drop it, fast-forward to the remote tip, and rebuild the union ledger from there.
            _git(["reset", "--hard", "origin/submission"], root)
            if attempt == 2:
                raise WorkflowError(
                    "origin/submission kept advancing during publication; the local branch "
                    "was reset to the remote tip safely. Rerun validation to publish the score."
                ) from push_error
            continue

        _git(["fetch", "origin", "submission"], root)
        pushed_keys = _remote_ledger_keys(root)
        if not old_keys.issubset(pushed_keys):
            raise WorkflowError(
                "remote leaderboard read-back is missing previously recorded entries"
            )
        if result_key(result) not in pushed_keys:
            raise WorkflowError("remote leaderboard read-back is missing the new submission")
        return

    raise WorkflowError("could not publish leaderboard after repeated concurrent updates")


def _refresh_missing_checks(pr: PullRequest) -> PullRequest:
    """Give delayed GitHub check events a short bounded window to appear."""
    if pr.merged or pr.checks_status != "none":
        return pr
    for delay in (1, 2):
        time.sleep(delay)
        pr = get_pull_request(pr.number)
        if pr.merged or pr.checks_status != "none":
            return pr
    return pr


def _is_merged_now(pr: PullRequest) -> PullRequest:
    if pr.merged:
        return pr
    try:
        latest = get_pull_request(pr.number)
    except GitHubError:
        return pr
    return latest if latest.merged else pr


def _verify_promoted(root: Path) -> None:
    _git(["fetch", "origin", "main", "submission"], root)
    if not _is_ancestor("origin/submission", "origin/main", root):
        raise WorkflowError(
            "promotion did not reach main: origin/main does not contain origin/submission"
        )
    main_keys = _remote_ledger_keys(root, "origin/main")
    submission_keys = _remote_ledger_keys(root)
    if not submission_keys.issubset(main_keys):
        missing = sorted(submission_keys - main_keys)
        raise WorkflowError(
            f"promotion did not carry every ledger entry to main. Missing keys: {missing[:8]}"
        )


def _promote(root: Path) -> None:
    """Idempotently promote submission to main, resuming after partial failures."""
    _git(["fetch", "origin", "main", "submission"], root)
    if _unpublished_score_commits(root):
        raise WorkflowError(
            "local submission has commit(s) that are not published to origin/submission; "
            "refusing to report promotion success. Resolve the publication failure and rerun."
        )
    if _is_ancestor("origin/submission", "origin/main", root):
        _verify_promoted(root)
        logger.info("All scored artifacts are already promoted to main")
        return
    if not _is_ancestor("origin/main", "origin/submission", root):
        raise WorkflowError(
            "origin/submission and origin/main have diverged; reconcile them with the normal PR flow"
        )
    ahead = int(_git(["rev-list", "--count", "origin/main..origin/submission"], root).strip())
    if ahead == 0:
        logger.info("All scored artifacts are already promoted to main")
        return

    number = find_promotion_pr()
    if number is None:
        number = create_promotion_pr()
        logger.info("Created promotion PR #%d", number)
    else:
        logger.info("Resuming promotion PR #%d", number)
    merge_promotion(number)
    _git(["fetch", "origin", "main"], root)
    _git(["merge", "--ff-only", "origin/main"], root)
    _git(["push", "origin", "submission"], root)
    _verify_promoted(root)
    logger.info("Promoted accepted submissions to main")


def run(root: Path, pr_number: int | None, data_root: Path) -> int:
    with _exclusive_run_lock(root):
        return _run(root, pr_number, data_root)


def _run(root: Path, pr_number: int | None, data_root: Path) -> int:
    branch = _git(["branch", "--show-current"], root).strip()
    if branch != "submission":
        raise WorkflowError(f"run from the submission branch, not {branch or 'detached HEAD'}")
    if _git(["status", "--porcelain", "--untracked-files=no"], root).strip():
        raise WorkflowError(
            "working tree has uncommitted tracked changes; commit or stash them first so the "
            "acceptance workflow cannot accidentally carry them into the leaderboard commit"
        )

    check_authentication()
    _sync_submission(root)
    prs = open_submission_prs()
    if pr_number is not None:
        prs = [pr for pr in prs if pr.number == pr_number]
        if not prs:
            # Merged PRs are deliberately not returned by `gh pr list --state open`.
            # Allow an explicit rerun to recover a merge whose ledger step failed.
            recovered_pr = get_pull_request(pr_number)
            if recovered_pr.base_ref != "submission" or not recovered_pr.merged:
                raise WorkflowError(
                    f"open PR {pr_number} targeting submission was not found; it is not a merged PR "
                    "that can be recovered"
                )
            prs = [recovered_pr]
    if not prs:
        logger.info("No open submission PRs found")

    accepted = 0
    skipped = 0
    for listed_pr in prs:
        pr = listed_pr
        directory = None
        result = None
        merged = pr.merged
        try:
            if not pr.merged and pr.checks_status == "none":
                pr = _refresh_missing_checks(pr)
            if not pr.merged and not pr.checks_passed:
                raise WorkflowError(
                    f"required GitHub checks are not successful ({pr.checks_summary}); "
                    "rerun once checks have completed"
                )

            _git(["fetch", "origin", "submission"], root)
            _git(
                [
                    "fetch",
                    "origin",
                    f"pull/{pr.number}/head:refs/remotes/origin/pr/{pr.number}",
                ],
                root,
            )
            directory = _validate_pr(pr, root, _current_validation_base(root))
            result = _score_candidate(pr, directory, root, root / data_root)

            if not pr.merged:
                # Do not use `--delete-branch`: gh may fail deleting a local branch that
                # is checked out in another worktree after the remote merge has succeeded.
                merge(pr.number, delete_branch=False)
                merged = True
            _git(["pull", "--ff-only", "origin", "submission"], root)
            _publish_result(root, directory, result, pr_number=pr.number)
            accepted += 1
            logger.info("Accepted PR #%d (%s)", pr.number, result.submission)

            try:
                delete_remote_branch(pr.number)
            except _RECOVERY_ERRORS as cleanup_error:
                # Branch cleanup is housekeeping, not part of accepting/scoring a model.
                logger.warning(
                    "Could not clean remote head branch for PR #%d: %s", pr.number, cleanup_error
                )
        except _RECOVERY_ERRORS as exc:
            current_pr = _is_merged_now(pr)
            merged = merged or current_pr.merged
            if merged:
                # A merged PR is never reported as an unaccepted submission. Try to finish
                # scoring/artifacts now; an explicit `--pr N` rerun can resume if this fails.
                try:
                    if directory is None:
                        _sync_submission(root)
                        directory = _validate_pr(current_pr, root, _current_validation_base(root))
                    if result is None:
                        result = _score_candidate(current_pr, directory, root, root / data_root)
                    _sync_submission(root)
                    _git(["pull", "--ff-only", "origin", "submission"], root)
                    _publish_result(root, directory, result, pr_number=pr.number)
                    accepted += 1
                    logger.warning("Recovered post-merge acceptance for PR #%d", pr.number)
                    continue
                except _RECOVERY_ERRORS as recovery_error:
                    skipped += 1
                    logger.error(
                        "PR #%d is already merged but scoring/ledger recovery still failed: %s. "
                        "Rerun `mnnc validate-submissions --pr %d` after fixing the cause.",
                        pr.number,
                        recovery_error,
                        pr.number,
                    )
                    continue

            skipped += 1
            message = f"Automated submission validation failed:\n\n{exc}"
            logger.error("PR #%d skipped: %s", pr.number, exc)
            comment(pr.number, message)

    logger.info(
        "Validation summary: %d processed, %d accepted, %d skipped", len(prs), accepted, skipped
    )
    # Promotion is an independent, idempotent stage: it also resumes a promotion
    # whose merge or final synchronisation failed on an earlier run.
    _promote(root)
    return 0
