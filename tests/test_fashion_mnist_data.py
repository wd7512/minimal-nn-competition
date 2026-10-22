import gzip
import struct
from pathlib import Path

import torch

from minimal_nn_competition.benchmarks.fashion_mnist import data_loader


def _image_bytes(images: list[list[int]], rows: int, columns: int) -> bytes:
    header = struct.pack(">IIII", 2051, len(images), rows, columns)
    return header + bytes(value for image in images for value in image)


def _label_bytes(labels: list[int]) -> bytes:
    return struct.pack(">II", 2049, len(labels)) + bytes(labels)


def _write_gzip(path: Path, payload: bytes) -> None:
    path.write_bytes(gzip.compress(payload))


def test_download_fashion_mnist_writes_caches(tmp_path: Path, monkeypatch) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write_gzip(fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[1, 2, 3, 4]], 2, 2))
    _write_gzip(fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([1]))
    _write_gzip(fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[5, 6, 7, 8]], 2, 2))
    _write_gzip(fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([2]))

    def fake_download(url: str, path: Path) -> None:
        path.write_bytes((fixtures / Path(url).name).read_bytes())

    monkeypatch.setattr(data_loader, "download_file", fake_download)
    destination = tmp_path / "data"
    train_path, test_path = data_loader.download_fashion_mnist(destination)

    train_inputs, train_labels = torch.load(train_path, weights_only=True)
    test_inputs, test_labels = torch.load(test_path, weights_only=True)
    assert train_inputs.tolist() == [[1, 2, 3, 4]]
    assert train_labels.tolist() == [1]
    assert test_inputs.tolist() == [[5, 6, 7, 8]]
    assert test_labels.tolist() == [2]
