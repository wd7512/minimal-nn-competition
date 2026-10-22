"""Download and cache the official MNIST dataset without torchvision."""

from __future__ import annotations

from pathlib import Path

from minimal_nn_competition.benchmarks.idx import download_file as _download
from minimal_nn_competition.benchmarks.idx import read_images as _read_images
from minimal_nn_competition.benchmarks.idx import read_labels as _read_labels

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "mnist-train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "mnist-train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "mnist-t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "mnist-t10k-labels-idx1-ubyte.gz",
}


def download_mnist(destination: Path, force: bool = False) -> tuple[Path, Path]:
    """Download MNIST and save raw ``(inputs, labels)`` caches as ``.pt`` files."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "mnist-train.pt"
    test_path = destination / "mnist-test.pt"
    if train_path.is_file() and test_path.is_file() and not force:
        return train_path, test_path

    for remote, local in FILES.items():
        _download(MNIST_BASE_URL + remote, destination / local)

    train = (
        _read_images(destination / FILES["train-images-idx3-ubyte.gz"]),
        _read_labels(destination / FILES["train-labels-idx1-ubyte.gz"]),
    )
    test = (
        _read_images(destination / FILES["t10k-images-idx3-ubyte.gz"]),
        _read_labels(destination / FILES["t10k-labels-idx1-ubyte.gz"]),
    )
    torch.save(train, train_path)
    torch.save(test, test_path)
    return train_path, test_path
