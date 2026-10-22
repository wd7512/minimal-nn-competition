"""Helpers for downloading and parsing IDX-format image datasets."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Any


def download_file(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` unless the file already exists."""
    if path.is_file():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def read_images(path: Path) -> Any:
    """Read an IDX image file as a ``(count, rows * columns)`` uint8 tensor."""
    import torch

    raw = gzip.decompress(path.read_bytes())
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected IDX image magic number in {path}: {magic}")
    pixels = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8)
    return pixels.reshape(count, rows * columns).clone()


def read_labels(path: Path) -> Any:
    """Read an IDX label file as a ``(count,)`` int64 tensor."""
    import torch

    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected IDX label magic number in {path}: {magic}")
    labels = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8)
    return labels.to(torch.int64).reshape(count).clone()
