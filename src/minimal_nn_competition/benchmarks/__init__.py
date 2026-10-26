"""One module per benchmark, plus a central registry and shared helpers."""

from minimal_nn_competition.benchmarks.registry import available_benchmarks, create_benchmark

__all__ = ["available_benchmarks", "create_benchmark"]
