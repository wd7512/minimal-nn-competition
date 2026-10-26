from __future__ import annotations

import pytest
import torch

from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.examples import snake_submission


class _CapturingEnvironment:
    _captured: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        _CapturingEnvironment._captured.append(dict(kwargs))
        self.snake = [(0, 0)]
        self.won = False
        self.food = None

    def reset(self) -> tuple[float, ...]:
        return (0.0,) * 24

    def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
        return (0.0,) * 24, 0.0, True


@pytest.mark.parametrize("benchmark_name", SNAKE_BENCHMARKS)
def test_snake_example_evaluate_uses_benchmark_move_cap(
    monkeypatch: pytest.MonkeyPatch, benchmark_name: str
) -> None:
    _CapturingEnvironment._captured = []
    import minimal_nn_competition.benchmarks.snake.environment as snake_environment

    monkeypatch.setattr(snake_environment, "SnakeEnvironment", _CapturingEnvironment)
    benchmark = create_benchmark(benchmark_name)
    policy = torch.nn.Linear(24, 4)
    policy.eval()

    lengths = snake_submission._evaluate(policy, torch.device("cpu"), (0, 1), benchmark)

    assert lengths == [1, 1]
    assert len(_CapturingEnvironment._captured) == 2
    assert all(item["max_moves"] == benchmark.max_moves for item in _CapturingEnvironment._captured)
    # The total-move cap and per-food stall rule remain distinct settings.
    assert benchmark.max_moves != benchmark.moves_per_food


@pytest.mark.parametrize("benchmark_name", SNAKE_BENCHMARKS)
def test_snake_example_evaluate_handles_full_board_state(
    monkeypatch: pytest.MonkeyPatch, benchmark_name: str
) -> None:
    class _FullBoardEnvironment(_CapturingEnvironment):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.won = True
            self.snake = [(0, 0), (1, 0)]

    import minimal_nn_competition.benchmarks.snake.environment as snake_environment

    monkeypatch.setattr(snake_environment, "SnakeEnvironment", _FullBoardEnvironment)
    benchmark = create_benchmark(benchmark_name)
    policy = torch.nn.Linear(24, 4)
    policy.eval()

    lengths = snake_submission._evaluate(policy, torch.device("cpu"), (0,), benchmark)

    assert len(lengths) == 1
    assert lengths[0] >= 1


def test_snake_example_evaluate_applies_per_food_stall_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environments: list[object] = []

    class _StallingEnvironment:
        def __init__(self, **kwargs: object) -> None:
            self._calls = 0
            self.snake = [(0, 0)]
            self.won = False
            self.food = (1, 0)
            environments.append(self)

        def reset(self) -> tuple[float, ...]:
            self._calls = 0
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
            self._calls += 1
            return (0.0,) * 24, 0.0, False

    import minimal_nn_competition.benchmarks.snake.environment as snake_environment

    monkeypatch.setattr(snake_environment, "SnakeEnvironment", _StallingEnvironment)
    benchmark = create_benchmark("snake-obstacles-8x8")
    policy = torch.nn.Linear(24, 4)
    policy.eval()

    lengths = snake_submission._evaluate(policy, torch.device("cpu"), (0,), benchmark)

    assert lengths == [1]
    assert environments[0]._calls == benchmark.moves_per_food + 1


def test_snake_example_evaluate_resets_stall_counter_after_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environments: list[object] = []

    class _GrowingThenStallingEnvironment:
        def __init__(self, **kwargs: object) -> None:
            self._calls = 0
            self.snake = [(0, 0)]
            self.won = False
            self.food = (1, 0)
            environments.append(self)

        def reset(self) -> tuple[float, ...]:
            self._calls = 0
            self.snake = [(0, 0)]
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
            self._calls += 1
            if self._calls == 1:
                self.snake.append((1, 0))
            return (0.0,) * 24, 0.0, False

    import minimal_nn_competition.benchmarks.snake.environment as snake_environment

    monkeypatch.setattr(snake_environment, "SnakeEnvironment", _GrowingThenStallingEnvironment)
    benchmark = create_benchmark("snake-obstacles-8x8")
    policy = torch.nn.Linear(24, 4)
    policy.eval()

    lengths = snake_submission._evaluate(policy, torch.device("cpu"), (0,), benchmark)

    assert lengths == [2]
    assert environments[0]._calls == benchmark.moves_per_food + 2
