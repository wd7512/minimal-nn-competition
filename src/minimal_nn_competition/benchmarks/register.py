"""Benchmark registry."""


def _factories():
    from minimal_nn_competition.benchmarks.chess_eval.benchmark import ChessEvalBenchmark
    from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import (
        FashionMnistBenchmark,
    )
    from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
    from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark

    return {
        "chess-eval": ChessEvalBenchmark,
        "fashion-mnist": FashionMnistBenchmark,
        "mnist": MnistBenchmark,
        "snake-simple-8x8": SnakeBenchmark,
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
