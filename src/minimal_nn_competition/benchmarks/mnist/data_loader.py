"""Download and cache the official MNIST dataset without torchvision."""

from minimal_nn_competition.benchmarks.idx import download_dataset
from minimal_nn_competition.benchmarks.idx import download_file as _download

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

MNIST_SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def download_mnist(destination, force=False):
    """Download MNIST and save one raw ``(inputs, labels)`` cache as a ``.pt`` file."""
    return download_dataset(
        destination, MNIST_BASE_URL, "mnist", "mnist", MNIST_SHA256, force, _download
    )
