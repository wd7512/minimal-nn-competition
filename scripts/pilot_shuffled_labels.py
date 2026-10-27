"""Prepare and report deterministic shuffled-label MNIST pilot datasets."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from minimal_nn_competition.benchmarks.common.idx import sha256_file
from minimal_nn_competition.benchmarks.fashion_mnist import FASHION_MNIST_SHA256
from minimal_nn_competition.benchmarks.mnist import MNIST_SHA256

SEED = 191
DATASETS = {
    "mnist": MNIST_SHA256,
    "fashion-mnist": FASHION_MNIST_SHA256,
}


def make_labels(labels, seed=SEED):
    """Return a permutation of source labels and fresh independent uniform labels."""
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = torch.from_numpy(rng.permutation(labels.numpy()).copy()).long()
    uniform = torch.from_numpy(rng.integers(0, 10, size=len(labels), dtype=np.int64))
    return shuffled, uniform


def prepare(data_root, output_root):
    """Write permuted-label and independent-uniform-label caches with reports."""
    reports = {}
    for dataset, checksums in DATASETS.items():
        prefix = "fashion" if dataset == "fashion-mnist" else "mnist"
        source = data_root / f"{dataset}.pt"
        inputs, labels = torch.load(source, map_location="cpu", weights_only=True)
        if len(inputs) != 70_000 or len(labels) != 70_000:
            raise ValueError(f"{dataset} source cache must contain exactly 70,000 examples")
        shuffled, independent = make_labels(labels)
        raw_hashes = {name: sha256_file(data_root / f"{prefix}-{name}") for name in checksums}
        if raw_hashes != checksums:
            raise ValueError(f"{dataset} raw source files do not match their pinned checksums")
        reports[dataset] = {
            "source": str(source),
            "source_cache_sha256": sha256_file(source),
            "raw_file_sha256": raw_hashes,
            "expected_raw_file_sha256": checksums,
            "ordering": "official training split followed by official test split",
            "seed": SEED,
            "prng": "NumPy Generator(PCG64), NumPy " + np.__version__,
            "shuffled_label_counts": torch.bincount(shuffled, minlength=10).tolist(),
            "uniform_label_counts": torch.bincount(independent, minlength=10).tolist(),
            "duplicate_input_groups": _duplicate_groups(inputs),
            "outputs": {},
        }
        for suffix, targets in (("shuffled", shuffled), ("uniform", independent)):
            path = output_root / f"{dataset}-{suffix}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((inputs, targets), path)
            counts = torch.bincount(targets, minlength=10)
            reports[dataset]["outputs"][suffix] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "majority_constant_accuracy": int(counts.max()) / len(targets),
            }
    logging.info("%s", json.dumps(reports, indent=2, sort_keys=True))
    return reports


def _duplicate_groups(inputs):
    groups = {}
    for row in inputs.numpy():
        image = row.tobytes()
        key = hashlib.sha256(image).digest()
        groups.setdefault(key, (image, 0))
        stored, count = groups[key]
        if stored != image:
            raise ValueError("SHA-256 collision while checking duplicate inputs")
        groups[key] = (image, count + 1)
    return sum(count > 1 for _, count in groups.values())


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/shuffled-labels"))
    args = parser.parse_args()
    prepare(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
