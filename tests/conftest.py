from __future__ import annotations

from typing import Any

import pytest

import minimal_nn_competition.__main__ as cli
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark

FAST_SEEDS = (0, 1, 2)


@pytest.fixture
def fast_snake_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fast_create(name: str, **kwargs: Any) -> Any:
        if name == "snake-simple-8x8":
            return SnakeBenchmark(seeds=FAST_SEEDS, **kwargs)
        return create_benchmark(name, **kwargs)

    monkeypatch.setattr(cli, "create_benchmark", fast_create)
