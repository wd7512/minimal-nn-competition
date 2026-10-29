"""Versioned native Gymnasium lunarlander benchmark."""

from minimal_nn_competition.benchmarks.common.control import ControlBenchmark


class LunarLanderBenchmark(ControlBenchmark):
    name = "lunarlander-gym-v1"
