import json
from pathlib import Path

import torch

from minimal_nn_competition.examples import fashion_mnist_submission, mnist_submission


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
