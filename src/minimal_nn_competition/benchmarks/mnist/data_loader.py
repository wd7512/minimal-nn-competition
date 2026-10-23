"""Download and cache the official MNIST dataset without torchvision."""

from minimal_nn_competition.benchmarks.idx import download_dataset
from minimal_nn_competition.benchmarks.idx import download_file as _download

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def download_mnist(destination, force=False):
    """Download MNIST and save one raw ``(inputs, labels)`` cache as a ``.pt`` file."""
    return download_dataset(destination, MNIST_BASE_URL, "mnist", "mnist", force, _download)
