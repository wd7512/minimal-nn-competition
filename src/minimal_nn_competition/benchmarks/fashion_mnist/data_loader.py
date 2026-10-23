"""Download and cache the official Fashion-MNIST dataset without torchvision."""

from minimal_nn_competition.benchmarks.idx import download_dataset, download_file

FASHION_MNIST_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


def download_fashion_mnist(destination, force=False):
    """Download Fashion-MNIST and save raw ``(inputs, labels)`` caches as ``.pt`` files."""
    return download_dataset(
        destination, FASHION_MNIST_BASE_URL, "fashion", "fashion-mnist", force, download_file
    )
