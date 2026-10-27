from __future__ import annotations

from typing import Any

import pytest

import minimal_nn_competition.__main__ as cli
from minimal_nn_competition.benchmarks.registry import create_benchmark
from minimal_nn_competition.benchmarks.snake_obstacles_8x8 import SnakeObstacles8x8Benchmark
from minimal_nn_competition.benchmarks.snake_simple_8x8 import SnakeSimple8x8Benchmark
from minimal_nn_competition.benchmarks.snake_simple_16x16 import SnakeSimple16x16Benchmark

FAST_SEEDS = (0, 1, 2)


@pytest.fixture
def fast_snake_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fast_create(name: str, **kwargs: Any) -> Any:
        if name == "snake-simple-8x8":
            return SnakeSimple8x8Benchmark(seeds=FAST_SEEDS, **kwargs)
        if name == "snake-simple-16x16":
            return SnakeSimple16x16Benchmark(seeds=FAST_SEEDS, **kwargs)
        if name == "snake-obstacles-8x8":
            return SnakeObstacles8x8Benchmark(seeds=FAST_SEEDS, **kwargs)
        return create_benchmark(name, **kwargs)

    monkeypatch.setattr(cli, "create_benchmark", fast_create)
