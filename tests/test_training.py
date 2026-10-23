import json
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.examples import (
    classifier_submission,
    fashion_mnist_submission,
    mnist_submission,
)


def _write_training_cache(path: Path) -> None:
    inputs = torch.zeros((4, 784), dtype=torch.uint8)
    for index in range(4):
        inputs[index, index] = 255
    torch.save((inputs, torch.arange(4)), path)


def _assert_written(output: Path, benchmark: str) -> None:
    config = json.loads((output / "config.json").read_text())
    assert config["benchmark"] == benchmark
    assert config["name"] == output.name
    assert config["author"] == "unknown"
    assert config["input_size"] == 784
    assert config["output_size"] == 10
    assert config["architecture"] == "mlp"
    assert config["hidden_sizes"] == [64]
    assert config["activation"] == "relu"
    assert config["normalisation"] is False
    assert (output / "model.pt").is_file()


def test_mnist_training_writes_submission(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist-train.pt"
    _write_training_cache(data_path)
    output = tmp_path / "mnist"
    mnist_submission.train(
        data_path,
        output,
        epochs=1,
        batch_size=2,
        device_name="cpu",
    )
    _assert_written(output, "mnist")


def test_fashion_mnist_training_writes_submission(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist-train.pt"
    _write_training_cache(data_path)
    output = tmp_path / "fashion-mnist"
    fashion_mnist_submission.train(
        data_path,
        output,
        epochs=1,
        batch_size=2,
        device_name="cpu",
    )
    _assert_written(output, "fashion-mnist")


def test_training_processes_final_partial_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "mnist-train.pt"
    inputs = torch.zeros((5, 784), dtype=torch.uint8)
    torch.save((inputs, torch.arange(5)), data_path)
    batch_sizes: list[int] = []
    build_model = classifier_submission.build_model

    def recording_model(*args: object, **kwargs: object) -> torch.nn.Module:
        model = build_model(*args, **kwargs)
        model.register_forward_pre_hook(
            lambda _model, values: batch_sizes.append(values[0].shape[0])
        )
        return model

    monkeypatch.setattr(classifier_submission, "build_model", recording_model)
    mnist_submission.train(data_path, tmp_path / "mnist", epochs=1, batch_size=2, device_name="cpu")
    assert batch_sizes == [2, 2, 1]


@pytest.mark.parametrize(
    ("module", "benchmark"),
    ((mnist_submission, "mnist"), (fashion_mnist_submission, "fashion-mnist")),
)
def test_example_cli_trains(tmp_path: Path, module: object, benchmark: str) -> None:
    data_path = tmp_path / f"{benchmark}-train.pt"
    _write_training_cache(data_path)
    output = tmp_path / benchmark
    assert (
        module.main(
            [
                "--data-path",
                str(data_path),
                "--output",
                str(output),
                "--epochs",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    _assert_written(output, benchmark)
