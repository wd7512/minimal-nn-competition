import hashlib
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks import shuffled_mnist
from minimal_nn_competition.benchmarks.registry import available_benchmarks, create_benchmark
from minimal_nn_competition.model.entry import LoadedSubmission, SubmissionError


@pytest.mark.parametrize(
    ("benchmark", "sha256"),
    [
        ("mnist-shuffled", "ec783f3bee54df90c63abf522174265fa90f01db49e234ce58db58eb09c72719"),
        (
            "fashion-mnist-shuffled",
            "d290faeac22226f9b8695b363b9d789b6f7138ec5f959d0465da9199c7cf9f68",
        ),
    ],
)
def test_versioned_label_vectors_are_frozen(benchmark: str, sha256: str) -> None:
    payload, actual = shuffled_mnist._label_bytes(benchmark)
    assert len(payload) == 70_000
    assert actual == sha256
    assert hashlib.sha256(payload).hexdigest() == sha256


def test_shuffled_benchmarks_are_registered_and_exhaustive(tmp_path: Path) -> None:
    assert {"mnist-shuffled", "fashion-mnist-shuffled"} <= set(available_benchmarks())
    inputs = torch.zeros((70_000, 784), dtype=torch.uint8)
    labels = torch.zeros(70_000, dtype=torch.int64)
    labels[-1] = 1
    data_path = tmp_path / "labels.pt"
    torch.save((inputs, labels), data_path)
    config = type(
        "Config",
        (),
        {
            "benchmark": "mnist-shuffled",
            "input_size": 784,
            "output_size": 10,
            "normalisation": False,
        },
    )()
    model = torch.nn.Linear(784, 10)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    loaded = LoadedSubmission(config, {}, model)
    score, passed, details = create_benchmark(
        "mnist-shuffled", data_path=data_path, batch_size=256
    ).evaluate(loaded)
    assert score == pytest.approx(69_999 / 70_000)
    assert passed is False
    assert details["samples"] == 70_000


def test_download_shuffled_dataset_uses_pinned_labels(tmp_path: Path, monkeypatch) -> None:
    inputs = torch.zeros((70_000, 784), dtype=torch.uint8)
    indices = torch.arange(70_000)
    inputs[:, 0] = indices % 256
    inputs[:, 1] = (indices // 256) % 256
    inputs[:, 2] = indices // 65_536
    source = tmp_path / "mnist.pt"
    torch.save((inputs, torch.zeros(70_000, dtype=torch.int64)), source)
    calls = []

    def load_verified_source(destination, force=False):
        calls.append(force)
        return source

    monkeypatch.setattr(shuffled_mnist, "download_mnist", load_verified_source)

    output = shuffled_mnist.download_shuffled_dataset("mnist-shuffled", tmp_path)
    saved_inputs, labels = torch.load(output, weights_only=True)
    expected = torch.frombuffer(
        bytearray(shuffled_mnist._label_bytes("mnist-shuffled")[0]), dtype=torch.uint8
    ).long()
    assert output == tmp_path / "mnist-shuffled.pt"
    assert torch.equal(saved_inputs, inputs)
    assert torch.equal(labels, expected)
    assert calls == [True]


def test_duplicate_images_must_have_consistent_labels(monkeypatch) -> None:
    def numpy_unavailable(*args, **kwargs):
        raise RuntimeError("Numpy is not available")

    monkeypatch.setattr(torch.Tensor, "numpy", numpy_unavailable)
    images = torch.tensor([[1, 2], [1, 2], [2, 1]], dtype=torch.uint8)
    shuffled_mnist.validate_label_consistency(images, torch.tensor([3, 3, 4]))
    with pytest.raises(ValueError, match="conflicting shuffled labels"):
        shuffled_mnist.validate_label_consistency(images, torch.tensor([3, 5, 4]))


def test_download_shuffled_cli_wires_benchmark(tmp_path: Path, monkeypatch, capsys) -> None:
    expected = tmp_path / "mnist-shuffled.pt"
    monkeypatch.setattr(
        shuffled_mnist,
        "download_shuffled_dataset",
        lambda name, destination: expected,
    )
    assert main(["download-mnist-shuffled", "--output", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(expected)


def test_shuffled_dataset_rejects_non_finite_model_scores(tmp_path: Path) -> None:
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    labels = torch.tensor([0, 1])
    data_path = tmp_path / "small.pt"
    torch.save((inputs, labels), data_path)
    config = type(
        "Config",
        (),
        {
            "benchmark": "mnist-shuffled",
            "input_size": 784,
            "output_size": 10,
            "normalisation": False,
        },
    )()
    model = torch.nn.Linear(784, 10)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(float("inf"))
    loaded = LoadedSubmission(config, {}, model)
    with pytest.raises(SubmissionError, match="non-finite"):
        create_benchmark("mnist-shuffled", data_path=data_path, expected_samples=2).evaluate(loaded)
