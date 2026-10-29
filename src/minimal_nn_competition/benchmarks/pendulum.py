"""Versioned native Gymnasium pendulum benchmark."""

from minimal_nn_competition.benchmarks.common.control import ControlBenchmark


class PendulumBenchmark(ControlBenchmark):
    name = "pendulum-gym-v1"
