"""8x8 obstacle-Snake benchmark: fill the board around blocked cells."""

from minimal_nn_competition.benchmarks.common.snake.benchmark import BaseSnakeBenchmark
from minimal_nn_competition.benchmarks.common.snake.layout import obstacles_for_seed

__all__ = ["SnakeObstacles8x8Benchmark"]


class SnakeObstacles8x8Benchmark(BaseSnakeBenchmark):
    name = "snake-obstacles-8x8"
    size = 8
    loop_on_full = True
    moves_per_food = 2 * 8 * 8
    max_moves_default = 16384

    def obstacles(self, seed):
        return obstacles_for_seed(self.size, seed)
