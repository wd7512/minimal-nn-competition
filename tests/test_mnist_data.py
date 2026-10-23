import gzip
import struct
from pathlib import Path

import torch

from minimal_nn_competition.benchmarks.fashion_mnist import data_loader as fashion_loader
from minimal_nn_competition.benchmarks.idx import read_idx
from minimal_nn_competition.benchmarks.mnist import data_loader


def _image_bytes(images: list[list[int]], rows: int, columns: int) -> bytes:
    header = struct.pack(">IIII", 2051, len(images), rows, columns)
    return header + bytes(value for image in images for value in image)


def _label_bytes(labels: list[int]) -> bytes:
    return struct.pack(">II", 2049, len(labels)) + bytes(labels)


def _write_gzip(path: Path, payload: bytes) -> None:
    path.write_bytes(gzip.compress(payload))


def test_parsers_read_idx_data(tmp_path: Path) -> None:
    images_path = tmp_path / "images.gz"
    labels_path = tmp_path / "labels.gz"
    _write_gzip(images_path, _image_bytes([[0, 1, 2, 3], [4, 5, 6, 7]], 2, 2))
    _write_gzip(labels_path, _label_bytes([3, 7]))

    images = read_idx(images_path)
    labels = read_idx(labels_path, True)

    assert images.shape == (2, 4)
    assert images.dtype == torch.uint8
    assert images[1].tolist() == [4, 5, 6, 7]
    assert labels.tolist() == [3, 7]


def test_download_mnist_writes_caches(tmp_path: Path, monkeypatch) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write_gzip(fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[1, 2, 3, 4]], 2, 2))
    _write_gzip(fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([1]))
    _write_gzip(fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[5, 6, 7, 8]], 2, 2))
    _write_gzip(fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([2]))

    def fake_download(url: str, path: Path) -> None:
        path.write_bytes((fixtures / Path(url).name).read_bytes())

    monkeypatch.setattr(data_loader, "_download", fake_download)
    destination = tmp_path / "data"
    train_path, test_path = data_loader.download_mnist(destination)

    train_inputs, train_labels = torch.load(train_path, weights_only=True)
    test_inputs, test_labels = torch.load(test_path, weights_only=True)
    assert train_inputs.tolist() == [[1, 2, 3, 4]]
    assert train_labels.tolist() == [1]
    assert test_inputs.tolist() == [[5, 6, 7, 8]]
    assert test_labels.tolist() == [2]


def test_image_caches_use_distinct_raw_files(tmp_path: Path, monkeypatch) -> None:
    mnist_fixtures = tmp_path / "mnist-fixtures"
    fashion_fixtures = tmp_path / "fashion-fixtures"
    mnist_fixtures.mkdir()
    fashion_fixtures.mkdir()
    _write_gzip(mnist_fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[1, 2, 3, 4]], 2, 2))
    _write_gzip(mnist_fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([1]))
    _write_gzip(mnist_fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[5, 6, 7, 8]], 2, 2))
    _write_gzip(mnist_fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([2]))
    _write_gzip(
        fashion_fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[9, 10, 11, 12]], 2, 2)
    )
    _write_gzip(fashion_fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([3]))
    _write_gzip(
        fashion_fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[13, 14, 15, 16]], 2, 2)
    )
    _write_gzip(fashion_fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([4]))

    def download_from(fixtures):
        def fake_download(url: str, path: Path) -> None:
            if not path.exists():
                path.write_bytes((fixtures / Path(url).name).read_bytes())

        return fake_download

    monkeypatch.setattr(data_loader, "_download", download_from(mnist_fixtures))
    monkeypatch.setattr(fashion_loader, "download_file", download_from(fashion_fixtures))
    destination = tmp_path / "data"
    mnist_train, mnist_test = data_loader.download_mnist(destination)
    fashion_train, fashion_test = fashion_loader.download_fashion_mnist(destination)

    assert torch.load(mnist_train, weights_only=True)[0].tolist() == [[1, 2, 3, 4]]
    assert torch.load(mnist_test, weights_only=True)[0].tolist() == [[5, 6, 7, 8]]
    assert torch.load(fashion_train, weights_only=True)[0].tolist() == [[9, 10, 11, 12]]
    assert torch.load(fashion_test, weights_only=True)[0].tolist() == [[13, 14, 15, 16]]
