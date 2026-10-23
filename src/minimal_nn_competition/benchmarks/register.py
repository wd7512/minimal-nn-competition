"""Benchmark registry."""


def _factories():
    from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import (
        FashionMnistBenchmark,
    )
    from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
    from minimal_nn_competition.benchmarks.snake.benchmark import (
        SnakeBenchmark,
        SnakeObstacles8x8Benchmark,
        SnakeSimple16x16Benchmark,
    )

    return {
        "fashion-mnist": FashionMnistBenchmark,
        "mnist": MnistBenchmark,
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
