"""Benchmark-aware scoring for one submission."""

from __future__ import annotations

from pathlib import Path

from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.leaderboard import load_results, save_results
from minimal_nn_competition.model.entry import Submission
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
