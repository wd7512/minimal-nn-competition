"""MNIST benchmark: classify the full test set, ranked by accuracy then size."""

from __future__ import annotations

from minimal_nn_competition.benchmarks.image_classification import (
    ImageClassificationBenchmark,
)


class MnistBenchmark(ImageClassificationBenchmark):
    name = "mnist"
    display_name = "MNIST"
