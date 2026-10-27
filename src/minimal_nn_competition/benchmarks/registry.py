"""Benchmark registry: canonical benchmark IDs, family groups, and factories.

The factory mapping in :func:`_factories` is canonical: every registered
benchmark ID appears there exactly once. The ``*_BENCHMARKS`` tuples assign
each ID to one family for display and grouping; they must partition the
factory keys exactly (enforced by test_benchmark_families_partition_registry).
The two chess tiers share one implementation module.
"""

CHESS_BENCHMARKS = ("chess-eval-10k", "chess-eval-100k")
IMAGE_BENCHMARKS = ("mnist", "fashion-mnist")
SNAKE_BENCHMARKS = ("snake-simple-8x8", "snake-simple-16x16", "snake-obstacles-8x8")
RL_BENCHMARKS = ("cartpole", "pendulum", "lunarlander")


def _factories():
    from minimal_nn_competition.benchmarks.cartpole import CartPoleBenchmark
    from minimal_nn_competition.benchmarks.chess_eval import (
        ChessEval10kBenchmark,
        ChessEval100kBenchmark,
    )
    from minimal_nn_competition.benchmarks.fashion_mnist import FashionMnistBenchmark
    from minimal_nn_competition.benchmarks.lunarlander import LunarLanderBenchmark
    from minimal_nn_competition.benchmarks.mnist import MnistBenchmark
    from minimal_nn_competition.benchmarks.pendulum import PendulumBenchmark
    from minimal_nn_competition.benchmarks.snake_obstacles_8x8 import SnakeObstacles8x8Benchmark
    from minimal_nn_competition.benchmarks.snake_simple_8x8 import SnakeSimple8x8Benchmark
    from minimal_nn_competition.benchmarks.snake_simple_16x16 import SnakeSimple16x16Benchmark

    return {
        "cartpole": CartPoleBenchmark,
        "chess-eval-10k": ChessEval10kBenchmark,
        "chess-eval-100k": ChessEval100kBenchmark,
        "fashion-mnist": FashionMnistBenchmark,
        "lunarlander": LunarLanderBenchmark,
        "mnist": MnistBenchmark,
        "pendulum": PendulumBenchmark,
        "snake-simple-8x8": SnakeSimple8x8Benchmark,
        "snake-simple-16x16": SnakeSimple16x16Benchmark,
        "snake-obstacles-8x8": SnakeObstacles8x8Benchmark,
    }


def available_benchmarks():
    return tuple(sorted(_factories()))


def create_benchmark(name, **kwargs):
    try:
        factory = _factories()[name]
    except KeyError as exc:
        choices = ", ".join(available_benchmarks())
        raise ValueError(f"Unknown benchmark {name!r}; choose from {choices}") from exc
    return factory(**kwargs)
