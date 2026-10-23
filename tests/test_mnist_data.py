import gzip
import struct
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.__main__ import main
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


def _fixtures(tmp_path: Path) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write_gzip(fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[1, 2, 3, 4]], 2, 2))
    _write_gzip(fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([1]))
    _write_gzip(fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[5, 6, 7, 8]], 2, 2))
    _write_gzip(fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([2]))
    return fixtures


def _fake_download(fixtures: Path):
    def download(url: str, path: Path) -> None:
        path.write_bytes((fixtures / Path(url).name).read_bytes())

    return download


def _recording_download(fixtures: Path, calls: list[str]):
    def download(url: str, path: Path) -> None:
        calls.append(Path(url).name)
        path.write_bytes((fixtures / Path(url).name).read_bytes())

    return download


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


def test_download_mnist_writes_cache(tmp_path: Path, monkeypatch) -> None:
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
    cache_path = data_loader.download_mnist(destination)

    inputs, labels = torch.load(cache_path, weights_only=True)
    assert cache_path == destination / "mnist.pt"
    assert inputs.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert labels.tolist() == [1, 2]


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
    mnist_path = data_loader.download_mnist(destination)
    fashion_path = fashion_loader.download_fashion_mnist(destination)

    assert mnist_path == destination / "mnist.pt"
    assert fashion_path == destination / "fashion-mnist.pt"
    assert torch.load(mnist_path, weights_only=True)[0].tolist() == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    assert torch.load(fashion_path, weights_only=True)[0].tolist() == [
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]


def test_download_mnist_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(data_loader, "_download", _recording_download(fixtures, calls))
    destination = tmp_path / "data"

    first = data_loader.download_mnist(destination)
    calls.clear()

    assert data_loader.download_mnist(destination) == first
    assert calls == []


def test_download_mnist_force_rebuilds_cache(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(data_loader, "_download", _recording_download(fixtures, calls))
    destination = tmp_path / "data"

    first = data_loader.download_mnist(destination)
    calls.clear()

    assert data_loader.download_mnist(destination, force=True) == first
    assert len(calls) == 4


def test_download_mnist_does_not_write_split_caches(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    monkeypatch.setattr(data_loader, "_download", _fake_download(fixtures))
    destination = tmp_path / "data"

    data_loader.download_mnist(destination)

    assert not (destination / "mnist-train.pt").exists()
    assert not (destination / "mnist-test.pt").exists()


def test_download_mnist_cli_prints_cache_path(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures = _fixtures(tmp_path)
    monkeypatch.setattr(data_loader, "_download", _fake_download(fixtures))
    destination = tmp_path / "data"

    assert main(["download-mnist", "--output", str(destination)]) == 0
    assert capsys.readouterr().out.strip() == str(destination / "mnist.pt")
