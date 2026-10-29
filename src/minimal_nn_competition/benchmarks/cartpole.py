"""Versioned native Gymnasium cartpole benchmark."""

from minimal_nn_competition.benchmarks.common.control import ControlBenchmark


class CartPoleBenchmark(ControlBenchmark):
    name = "cartpole-gym-v1"
