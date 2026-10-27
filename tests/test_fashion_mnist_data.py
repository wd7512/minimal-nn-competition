import gzip
import hashlib
import struct
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks import fashion_mnist as data_loader
from minimal_nn_competition.benchmarks.common.idx import FILES
from minimal_nn_competition.benchmarks.fashion_mnist import FASHION_MNIST_SHA256


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


def _checksums_for(fixtures: Path) -> dict[str, str]:
    return {name: hashlib.sha256((fixtures / name).read_bytes()).hexdigest() for name in FILES}


def test_download_fashion_mnist_writes_cache(tmp_path: Path, monkeypatch) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write_gzip(fixtures / "train-images-idx3-ubyte.gz", _image_bytes([[1, 2, 3, 4]], 2, 2))
    _write_gzip(fixtures / "train-labels-idx1-ubyte.gz", _label_bytes([1]))
    _write_gzip(fixtures / "t10k-images-idx3-ubyte.gz", _image_bytes([[5, 6, 7, 8]], 2, 2))
    _write_gzip(fixtures / "t10k-labels-idx1-ubyte.gz", _label_bytes([2]))

    def fake_download(url: str, path: Path) -> None:
        path.write_bytes((fixtures / Path(url).name).read_bytes())

    monkeypatch.setattr(data_loader, "download_file", fake_download)
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", _checksums_for(fixtures))
    destination = tmp_path / "data"
    cache_path = data_loader.download_fashion_mnist(destination)

    inputs, labels = torch.load(cache_path, weights_only=True)
    assert cache_path == destination / "fashion-mnist.pt"
    assert inputs.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert labels.tolist() == [1, 2]


def test_pinned_fashion_mnist_checksums_cover_idx_files() -> None:
    assert set(FASHION_MNIST_SHA256) == set(FILES)
    assert all(
        len(digest) == 64 and int(digest, 16) >= 0 for digest in FASHION_MNIST_SHA256.values()
    )


def test_download_fashion_mnist_rejects_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    monkeypatch.setattr(data_loader, "download_file", _fake_download(fixtures))
    checksums = _checksums_for(fixtures)
    checksums["t10k-labels-idx1-ubyte.gz"] = "0" * 64
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", checksums)
    destination = tmp_path / "data"

    with pytest.raises(ValueError, match="Checksum mismatch for t10k-labels-idx1-ubyte.gz"):
        data_loader.download_fashion_mnist(destination)

    assert not (destination / "fashion-t10k-labels-idx1-ubyte.gz").exists()
    assert (destination / "fashion-train-images-idx3-ubyte.gz").exists()
    assert not (destination / "fashion-mnist.pt").exists()


def test_download_fashion_mnist_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(data_loader, "download_file", _recording_download(fixtures, calls))
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", _checksums_for(fixtures))
    destination = tmp_path / "data"

    first = data_loader.download_fashion_mnist(destination)
    calls.clear()

    assert data_loader.download_fashion_mnist(destination) == first
    assert calls == []


def test_download_fashion_mnist_force_rebuilds_cache(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(data_loader, "download_file", _recording_download(fixtures, calls))
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", _checksums_for(fixtures))
    destination = tmp_path / "data"

    first = data_loader.download_fashion_mnist(destination)
    calls.clear()

    assert data_loader.download_fashion_mnist(destination, force=True) == first
    assert len(calls) == 4


def test_download_fashion_mnist_does_not_write_split_caches(tmp_path: Path, monkeypatch) -> None:
    fixtures = _fixtures(tmp_path)
    monkeypatch.setattr(data_loader, "download_file", _fake_download(fixtures))
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", _checksums_for(fixtures))
    destination = tmp_path / "data"

    data_loader.download_fashion_mnist(destination)

    assert not (destination / "fashion-mnist-train.pt").exists()
    assert not (destination / "fashion-mnist-test.pt").exists()


def test_download_fashion_mnist_cli_prints_cache_path(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures = _fixtures(tmp_path)
    monkeypatch.setattr(data_loader, "download_file", _fake_download(fixtures))
    monkeypatch.setattr(data_loader, "FASHION_MNIST_SHA256", _checksums_for(fixtures))
    destination = tmp_path / "data"

    assert main(["download-fashion-mnist", "--output", str(destination)]) == 0
    assert capsys.readouterr().out.strip() == str(destination / "fashion-mnist.pt")
