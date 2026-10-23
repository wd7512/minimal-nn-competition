"""Helpers for downloading and parsing IDX-format image datasets."""

import gzip
import struct
import urllib.request

FILES = """train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz""".splitlines()


def download_file(url, path):
    """Download ``url`` to ``path`` unless the file already exists."""
    if path.is_file():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def read_idx(path, labels=False):
    """Read an IDX file as an image matrix or a vector of int64 labels."""
    import torch

    raw = gzip.decompress(path.read_bytes())
    fmt, magic = (">II", 2049) if labels else (">IIII", 2051)
    header = struct.unpack_from(fmt, raw)
    if header[0] != magic:
        kind = "label" if labels else "image"
        raise ValueError(f"Unexpected IDX {kind} magic number in {path}: {header[0]}")
    values = torch.frombuffer(bytearray(raw[struct.calcsize(fmt) :]), dtype=torch.uint8)
    if labels:
        return values.to(torch.int64).reshape(header[1]).clone()
    return values.reshape(header[1], header[2] * header[3]).clone()


def download_dataset(destination, base_url, prefix, cache, force=False, download=download_file):
    """Download IDX files into raw caches and save ``(inputs, labels)`` tensors."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / f"{cache}-train.pt"
    test_path = destination / f"{cache}-test.pt"
    if train_path.is_file() and test_path.is_file() and not force:
        return train_path, test_path
    train_images, train_labels, test_images, test_labels = (
        destination / f"{prefix}-{name}" for name in FILES
    )
    for name in FILES:
        download(base_url + name, destination / f"{prefix}-{name}")
    train = (read_idx(train_images), read_idx(train_labels, True))
    test = (read_idx(test_images), read_idx(test_labels, True))
    torch.save(train, train_path)
    torch.save(test, test_path)
    return train_path, test_path
