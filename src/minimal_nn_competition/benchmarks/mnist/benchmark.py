"""MNIST benchmark: classify the full dataset, ranked by accuracy then size."""

from minimal_nn_competition.benchmarks.image_classification import ImageClassificationBenchmark


class MnistBenchmark(ImageClassificationBenchmark):
    name = "mnist"
    display_name = "MNIST"
