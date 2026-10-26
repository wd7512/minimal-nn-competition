"""Benchmark-aware scoring for one submission."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.leaderboard import load_results, save_results
from minimal_nn_competition.model.entry import Submission, SubmissionError
from minimal_nn_competition.runner.entry import evaluate_submission


def score_submission(root: Path, directory: Path, data_root: Path):
    submission = Submission.from_directory(root / directory)
    kwargs = {}
    if submission.config.benchmark == "mnist":
        from minimal_nn_competition.benchmarks.mnist.data_loader import download_mnist

        kwargs["data_path"] = download_mnist(data_root)
    elif submission.config.benchmark == "fashion-mnist":
        from minimal_nn_competition.benchmarks.fashion_mnist.data_loader import (
            download_fashion_mnist,
        )

        kwargs["data_path"] = download_fashion_mnist(data_root)
    benchmark = create_benchmark(submission.config.benchmark, **kwargs)
    return evaluate_submission(submission, benchmark)


def result_key(result) -> tuple[str, str]:
    """Return the stable identity of a leaderboard row."""
    return result.benchmark, result.submission


def upsert_result(path: Path, result) -> None:
    """Replace only the matching benchmark/submission pair, preserving other rows."""
    results = load_results(path) if path.exists() else []
    key = result_key(result)
    results = [item for item in results if result_key(item) != key]
    results.append(result)
    save_results(path, results)


DEFAULT_SCORING_TIMEOUT_SECONDS = 600


def _score_worker(queue, root, directory, data_root):
    try:
        result = score_submission(Path(root), Path(directory), Path(data_root))
        queue.put(("ok", result))
    except BaseException as exc:  # Report any child failure to the parent process.
        queue.put(("error", repr(exc)))


def score_submission_process(
    root,
    directory,
    data_root,
    *,
    timeout=DEFAULT_SCORING_TIMEOUT_SECONDS,
):
    """Evaluate a submission in a disposable, timeout-bounded child process."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_score_worker,
        args=(queue, str(root), str(directory), str(data_root)),
        daemon=True,
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise SubmissionError(f"Scoring timed out after {timeout} seconds")
    try:
        status, payload = queue.get(timeout=5)
    except Exception as exc:
        raise SubmissionError(
            f"Scoring process exited with code {process.exitcode} and no result: {exc}"
        ) from exc
    if status == "error":
        raise SubmissionError(f"Scoring failed in isolated process: {payload}")
    return payload
