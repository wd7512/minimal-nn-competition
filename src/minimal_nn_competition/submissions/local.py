"""Local submission validation with explicit artifact updates."""

import logging
import shutil
import tempfile
from pathlib import Path

from minimal_nn_competition.benchmarks.registry import (
    RL_BENCHMARKS,
    SNAKE_BENCHMARKS,
    available_benchmarks,
    create_benchmark,
)
from minimal_nn_competition.leaderboard import load_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission, SubmissionError
from minimal_nn_competition.submissions.scoring import score_submission_process, upsert_result
from minimal_nn_competition.submissions.validation import validate_config

logger = logging.getLogger(__name__)


def validate_directory(directory: Path) -> Submission:
    """Check the folder and ensure its benchmark agrees with its location."""
    submission = Submission.from_directory(directory)
    benchmark = submission.config.benchmark
    benchmarks = available_benchmarks()
    if benchmark not in benchmarks:
        raise SubmissionError(f"Unknown benchmark: {benchmark}")
    parent = directory.parent
    if (parent.name in benchmarks or parent.parent.name == "models") and parent.name != benchmark:
        raise SubmissionError(
            f"Folder benchmark {parent.name!r} does not match config benchmark {benchmark!r}"
        )
    errors = validate_config(
        submission.config.model_dump(mode="json"), Path("models") / benchmark / directory.name
    )
    if errors:
        raise SubmissionError("\n".join(errors))
    return submission


def update_artifacts(directory: Path, submission: Submission, result, output: Path) -> None:
    """Prepare all artifacts before installing them, preserving existing results."""
    from minimal_nn_competition.submissions.artifacts import (
        _load_replays,
        _record_replay,
        install_site,
    )

    with tempfile.TemporaryDirectory(prefix="mnnc-update-") as temp:
        staged = Path(temp) / "docs"
        if output.exists():
            shutil.copytree(output, staged)
        else:
            staged.mkdir()
        ledger = staged / "leaderboard.json"
        upsert_result(ledger, result)
        if result.benchmark in SNAKE_BENCHMARKS + RL_BENCHMARKS:
            _record_replay(
                staged / "games", directory.name, create_benchmark(result.benchmark), submission
            )
        render_site(load_results(ledger), staged, _load_replays(staged / "games"))
        install_site(staged, output)


def run(directory: Path, data_root: Path, *, update: bool = False) -> int:
    directory = directory.resolve()
    submission = validate_directory(directory)
    result = score_submission_process(directory, data_root.resolve())
    logger.info("Validation report: %s", result.to_json())
    if update:
        update_artifacts(directory, submission, result, Path.cwd() / "docs")
        logger.info("Updated local leaderboard, replays, and HTML in docs/")
    return 0 if result.passed else 1
