"""Download and cache the official MNIST dataset without torchvision."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, path: Path) -> None:
    if path.is_file():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _read_images(path: Path):
    import torch

    raw = gzip.decompress(path.read_bytes())
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")
    pixels = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8)
    return pixels.reshape(count, rows * columns).clone()


def _read_labels(path: Path):
    import torch

    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")
    labels = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8)
    return labels.to(torch.int64).reshape(count).clone()


def download_mnist(destination: Path, force: bool = False) -> tuple[Path, Path]:
    """Download MNIST and save raw ``(inputs, labels)`` caches as ``.pt`` files."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "mnist-train.pt"
    test_path = destination / "mnist-test.pt"
    if train_path.is_file() and test_path.is_file() and not force:
        return train_path, test_path

    for filename in FILES:
        _download(MNIST_BASE_URL + filename, destination / filename)

    train = (
        _read_images(destination / "train-images-idx3-ubyte.gz"),
        _read_labels(destination / "train-labels-idx1-ubyte.gz"),
    )
    test = (
        _read_images(destination / "t10k-images-idx3-ubyte.gz"),
        _read_labels(destination / "t10k-labels-idx1-ubyte.gz"),
    )
    torch.save(train, train_path)
    torch.save(test, test_path)
    return train_path, test_path
