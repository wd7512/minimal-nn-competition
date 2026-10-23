"""Benchmark registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

BenchmarkFactory = Callable[..., Any]


def _factories() -> dict[str, BenchmarkFactory]:
    from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import (
        FashionMnistBenchmark,
    )
    from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
    from minimal_nn_competition.benchmarks.robustness.benchmark import RobustnessBenchmark
    from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark

    return {
        "fashion-mnist": FashionMnistBenchmark,
        "mnist": MnistBenchmark,
        "robustness": RobustnessBenchmark,
        "snake": SnakeBenchmark,
    }


def available_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(_factories()))


def create_benchmark(name: str, **kwargs: Any) -> Any:
    try:
        factory = _factories()[name]
    except KeyError as exc:
        choices = ", ".join(available_benchmarks())
        raise ValueError(f"Unknown benchmark {name!r}; choose from {choices}") from exc
    return factory(**kwargs)
