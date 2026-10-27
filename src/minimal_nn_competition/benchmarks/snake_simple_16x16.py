"""16x16 plain-Snake benchmark: fill the larger board, ranked by minimum parameters."""

from minimal_nn_competition.benchmarks.common.snake.benchmark import BaseSnakeBenchmark

__all__ = ["SnakeSimple16x16Benchmark"]


class SnakeSimple16x16Benchmark(BaseSnakeBenchmark):
    name = "snake-simple-16x16"
    size = 16
    # Correct play needs ~16 ticks/food, not 256: cap each feed at 2x the
    # board and lift the total backstop so the clock never decides a fill.
    moves_per_food = 2 * 16 * 16
    max_moves_default = 65536
