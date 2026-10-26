"""8x8 plain-Snake benchmark: fill the board, ranked by minimum parameters."""

from minimal_nn_competition.benchmarks.common.snake.benchmark import (
    DEFAULT_SEEDS,
    BaseSnakeBenchmark,
    selected_input_indices,
)

__all__ = ["DEFAULT_SEEDS", "SnakeBenchmark", "selected_input_indices"]


class SnakeBenchmark(BaseSnakeBenchmark):
    name = "snake-simple-8x8"
    size = 8
