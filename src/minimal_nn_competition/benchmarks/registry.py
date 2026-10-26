"""Benchmark registry."""


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
    from minimal_nn_competition.benchmarks.snake_simple_8x8 import SnakeBenchmark
    from minimal_nn_competition.benchmarks.snake_simple_16x16 import SnakeSimple16x16Benchmark

    return {
        "cartpole": CartPoleBenchmark,
        "chess-eval-10k": ChessEval10kBenchmark,
        "chess-eval-100k": ChessEval100kBenchmark,
        "fashion-mnist": FashionMnistBenchmark,
        "lunarlander": LunarLanderBenchmark,
        "mnist": MnistBenchmark,
        "pendulum": PendulumBenchmark,
        "snake-simple-8x8": SnakeBenchmark,
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
