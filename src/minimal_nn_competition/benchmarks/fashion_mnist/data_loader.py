"""Download and cache the official Fashion-MNIST dataset without torchvision."""

from __future__ import annotations

from pathlib import Path

from minimal_nn_competition.benchmarks.idx import download_file, read_images, read_labels

FASHION_MNIST_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "fashion-train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "fashion-train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "fashion-t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "fashion-t10k-labels-idx1-ubyte.gz",
}


def download_fashion_mnist(destination: Path, force: bool = False) -> tuple[Path, Path]:
    """Download Fashion-MNIST and save raw ``(inputs, labels)`` caches as ``.pt`` files."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "fashion-mnist-train.pt"
    test_path = destination / "fashion-mnist-test.pt"
    if train_path.is_file() and test_path.is_file() and not force:
        return train_path, test_path

    for remote, local in FILES.items():
        download_file(FASHION_MNIST_BASE_URL + remote, destination / local)

    train = (
        read_images(destination / FILES["train-images-idx3-ubyte.gz"]),
        read_labels(destination / FILES["train-labels-idx1-ubyte.gz"]),
    )
    test = (
        read_images(destination / FILES["t10k-images-idx3-ubyte.gz"]),
        read_labels(destination / FILES["t10k-labels-idx1-ubyte.gz"]),
    )
    torch.save(train, train_path)
    torch.save(test, test_path)
    return train_path, test_path
