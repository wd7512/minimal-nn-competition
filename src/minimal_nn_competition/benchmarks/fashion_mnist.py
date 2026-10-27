"""Fashion-MNIST benchmark: classify the full dataset, ranked by accuracy then size."""

from minimal_nn_competition.benchmarks.common.idx import download_dataset, download_file
from minimal_nn_competition.benchmarks.common.image_classification import (
    ImageClassificationBenchmark,
)

__all__ = [
    "FASHION_MNIST_BASE_URL",
    "FASHION_MNIST_SHA256",
    "FashionMnistBenchmark",
    "download_fashion_mnist",
]

FASHION_MNIST_BASE_URL = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/"
    "b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/"
)

FASHION_MNIST_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


class FashionMnistBenchmark(ImageClassificationBenchmark):
    name = "fashion-mnist"
    display_name = "Fashion-MNIST"


def download_fashion_mnist(destination, force=False):
    """Download Fashion-MNIST and save one raw ``(inputs, labels)`` cache as a ``.pt`` file."""
    return download_dataset(
        destination,
        FASHION_MNIST_BASE_URL,
        "fashion",
        "fashion-mnist",
        FASHION_MNIST_SHA256,
        force,
        download_file,
    )
