"""Shuffled-label variants of the public MNIST image datasets."""

import hashlib
import logging
from pathlib import Path

from minimal_nn_competition.benchmarks.common.image_classification import (
    ImageClassificationBenchmark,
)
from minimal_nn_competition.benchmarks.fashion_mnist import download_fashion_mnist
from minimal_nn_competition.benchmarks.mnist import download_mnist

logger = logging.getLogger(__name__)
_LABEL_ROOT = Path(__file__).with_name("data")
_LABELS = {
    "mnist-shuffled": (
        "mnist",
        "mnist-shuffled-v1.bin",
        "ec783f3bee54df90c63abf522174265fa90f01db49e234ce58db58eb09c72719",
    ),
    "fashion-mnist-shuffled": (
        "fashion-mnist",
        "fashion-mnist-shuffled-v1.bin",
        "d290faeac22226f9b8695b363b9d789b6f7138ec5f959d0465da9199c7cf9f68",
    ),
}


def _label_bytes(name):
    dataset, filename, expected = _LABELS[name]
    payload = (_LABEL_ROOT / filename).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected and actual != expected:
        raise ValueError(f"{dataset} shuffled-label checksum mismatch: {actual}")
    if len(payload) != 70_000:
        raise ValueError(f"{dataset} shuffled labels must contain exactly 70,000 values")
    return payload, actual


def download_shuffled_dataset(name, destination):
    """Download verified source images and combine them with the frozen v1 labels."""
    import torch

    # Bypass any existing .pt cache; each loader verifies raw hashes and rebuilds it.
    if name == "mnist-shuffled":
        source_path = download_mnist(destination, force=True)
    elif name == "fashion-mnist-shuffled":
        source_path = download_fashion_mnist(destination, force=True)
    else:
        raise ValueError(f"Unknown shuffled-label benchmark {name!r}")
    payload, label_sha256 = _label_bytes(name)
    inputs, _ = torch.load(source_path, map_location="cpu", weights_only=True)
    if inputs.shape != (70_000, 784):
        raise ValueError(f"{name} source images must have shape (70000, 784)")
    labels = torch.frombuffer(bytearray(payload), dtype=torch.uint8).long()
    validate_label_consistency(inputs, labels)
    output = Path(destination) / f"{name}.pt"
    torch.save((inputs, labels), output)
    logger.info("Wrote %s (labels SHA-256 %s)", output, label_sha256)
    return output


def validate_label_consistency(inputs, labels):
    """Reject duplicate pixel vectors assigned different fixed labels."""
    import torch

    seen = {}
    for image, label in zip(inputs, labels, strict=True):
        # Convert the uint8 pixel tensor directly; numpy is a dev-only dependency.
        raw = bytes(image.contiguous().view(torch.uint8).tolist())
        key = hashlib.sha256(raw).digest()
        prior = seen.get(key)
        if prior is not None:
            prior_image, prior_label = prior
            if prior_image != raw:
                raise ValueError("SHA-256 collision while checking duplicate images")
            if prior_label != int(label):
                raise ValueError("duplicate images have conflicting shuffled labels")
        else:
            seen[key] = (raw, int(label))


class MnistShuffledBenchmark(ImageClassificationBenchmark):
    name = "mnist-shuffled"
    display_name = "MNIST shuffled labels"


class FashionMnistShuffledBenchmark(ImageClassificationBenchmark):
    name = "fashion-mnist-shuffled"
    display_name = "Fashion-MNIST shuffled labels"
