"""Helpers for downloading and parsing IDX-format image datasets."""

import gzip
import hashlib
import struct
import urllib.request

FILES = """train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz""".splitlines()


def sha256_file(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def download_dataset(
    destination, base_url, prefix, cache, checksums, force=False, download=download_file
):
    """Download IDX files, verify their SHA-256 digests, and save one ``(inputs, labels)`` cache.

    ``checksums`` maps every name in ``FILES`` to its pinned SHA-256 hex digest.
    Each raw file is verified after download (or when already present on disk);
    a mismatch deletes the file and raises. The ``.pt`` cache short-circuits
    before any download, so it is only as trustworthy as the verified files it
    was built from.
    """
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / f"{cache}.pt"
    if cache_path.is_file() and not force:
        return cache_path
    train_images, train_labels, test_images, test_labels = (
        destination / f"{prefix}-{name}" for name in FILES
    )
    for name in FILES:
        path = destination / f"{prefix}-{name}"
        download(base_url + name, path)
        actual = sha256_file(path)
        if actual != checksums[name]:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {name}: expected {checksums[name]}, got {actual}; "
                "removed the bad file, retry the download"
            )
    inputs = torch.cat((read_idx(train_images), read_idx(test_images)))
    labels = torch.cat((read_idx(train_labels, True), read_idx(test_labels, True)))
    if len(inputs) != len(labels):
        raise ValueError(f"{cache} inputs and labels must have the same length")
    torch.save((inputs, labels), cache_path)
    return cache_path
