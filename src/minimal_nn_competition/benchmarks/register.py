"""Benchmark registry."""


def _factories():
    from minimal_nn_competition.benchmarks.cartpole.benchmark import CartPoleBenchmark
    from minimal_nn_competition.benchmarks.chess_eval.benchmark import (
        ChessEval10kBenchmark,
        ChessEval100kBenchmark,
    )
    from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import (
        FashionMnistBenchmark,
    )
    from minimal_nn_competition.benchmarks.lunarlander.benchmark import LunarLanderBenchmark
    from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
    from minimal_nn_competition.benchmarks.pendulum.benchmark import PendulumBenchmark
    from minimal_nn_competition.benchmarks.snake.benchmark import (
        SnakeBenchmark,
        SnakeObstacles8x8Benchmark,
        SnakeSimple16x16Benchmark,
    )

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
