import json
from pathlib import Path
from typing import Any

import pytest
import torch

from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import FashionMnistBenchmark
from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
from minimal_nn_competition.benchmarks.snake import benchmark as snake_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark
from minimal_nn_competition.model.entry import (
    Submission,
    SubmissionError,
    materialize_submission,
)


def _write_submission(
    tmp_path: Path, config: dict[str, Any], state: dict[str, torch.Tensor]
) -> Submission:
    directory = tmp_path / "example"
    directory.mkdir()
    config = {"name": "example", "author": "tester", **config}
    (directory / "config.json").write_text(json.dumps(config))
    torch.save(state, directory / "model.pt")
    return Submission.from_directory(directory)


def test_snake_partial_run_does_not_pass(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    result = SnakeBenchmark(seeds=(0, 1, 2)).evaluate(materialize_submission(submission))
    score, score_std, passed, details = result
    assert score == 1.0
    assert score_std == 0.0
    assert passed is False
    assert details["seed_lengths"] == [1, 1, 1]
    assert details["seed_moves"] == [5, 5, 5]


def test_snake_full_board_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FullBoardEnvironment:
        size = 8

        def __init__(
            self,
            size: int,
            seed: int,
            max_moves: int,
            input_indices: object = None,
            blocked: object = (),
            loop_on_full: bool = False,
        ) -> None:
            self.snake = [(0, 0)] * 64

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", FullBoardEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeBenchmark(seeds=(0, 1, 2)).evaluate(
        materialize_submission(submission)
    )
    assert score == 64.0
    assert score_std == 0.0
    assert passed is True
    assert details["seed_lengths"] == [64, 64, 64]
    assert details["seed_moves"] == [1, 1, 1]


def test_snake_defaults_to_100_seeds() -> None:
    assert SnakeBenchmark().seeds == tuple(range(100))


def test_snake_score_std_across_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class VariableEnvironment:
        size = 8

        def __init__(
            self,
            size: int,
            seed: int,
            max_moves: int,
            input_indices: object = None,
            blocked: object = (),
            loop_on_full: bool = False,
        ) -> None:
            self.snake = [(0, 0)] * (64 if seed == 0 else 1)

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", VariableEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeBenchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert score == 32.5
    assert score_std == 31.5
    assert passed is False
    assert details["seed_lengths"] == [64, 1]
    assert details["seed_moves"] == [1, 1]


def test_snake_rejects_wrong_config(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 8,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 8)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_size 24"):
        SnakeBenchmark().load_submission(submission)


def test_snake_selected_input_indices_build_smaller_model(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 5,
            "output_size": 4,
            "input_indices": [0, 1, 2, 4, 7],
        },
        {"weight": torch.zeros((4, 5)), "bias": torch.zeros(4)},
    )
    loaded = SnakeBenchmark().load_submission(submission)
    assert loaded.model.in_features == 5


def test_snake_selected_inputs_evaluate(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 5,
            "output_size": 4,
            "input_indices": [0, 1, 2, 4, 7],
        },
        {"weight": torch.zeros((4, 5)), "bias": torch.zeros(4)},
    )
    result = SnakeBenchmark(seeds=(0,)).evaluate(materialize_submission(submission))
    score, score_std, passed, details = result
    assert score == 1.0
    assert score_std == 0.0
    assert passed is False
    assert details["seed_lengths"] == [1]


def test_snake_rejects_out_of_range_input_indices(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 1,
            "output_size": 4,
            "input_indices": [24],
        },
        {"weight": torch.zeros((4, 1)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_indices"):
        SnakeBenchmark().load_submission(submission)


def _write_mnist_cache(path: Path) -> None:
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    inputs[0, 0] = 255
    inputs[1, 1] = 255
    torch.save((inputs, torch.tensor([0, 1])), path)


def test_mnist_evaluation_and_threshold(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = MnistBenchmark(
        data_path=data_path, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}

    score, _, passed, _ = MnistBenchmark(
        data_path=data_path, threshold=1.01, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert passed is False


def test_mnist_normalisation_flag(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
            "normalisation": True,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = MnistBenchmark(
        data_path=data_path, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": True}


def _write_mnist_cache_with(path: Path, inputs: torch.Tensor, labels: torch.Tensor) -> None:
    torch.save((inputs, labels), path)


def _loaded_mnist_submission(tmp_path: Path) -> object:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    return loaded


def _loaded_fashion_mnist_submission(tmp_path: Path, normalisation: bool = False) -> object:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
            "normalisation": normalisation,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    return loaded


def test_mnist_accepts_two_dimensional_labels(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    inputs[0, 0] = 255
    inputs[1, 1] = 255
    _write_mnist_cache_with(data_path, inputs, torch.tensor([[0], [1]]))
    loaded = _loaded_mnist_submission(tmp_path)
    score, _, passed, details = MnistBenchmark(data_path=data_path, expected_samples=2).evaluate(
        loaded
    )
    assert score == 1.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}


def test_mnist_rejects_length_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    _write_mnist_cache_with(
        data_path, torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1, 2])
    )
    loaded = _loaded_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="must contain raw"):
        MnistBenchmark(data_path=data_path, expected_samples=2).evaluate(loaded)


def test_mnist_rejects_wrong_input_width(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    _write_mnist_cache_with(
        data_path, torch.zeros((2, 10), dtype=torch.uint8), torch.tensor([0, 1])
    )
    loaded = _loaded_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="must contain raw"):
        MnistBenchmark(data_path=data_path, expected_samples=2).evaluate(loaded)


def test_mnist_rejects_partial_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    _write_mnist_cache(data_path)
    loaded = _loaded_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="exactly 70000 samples, found 2"):
        MnistBenchmark(data_path=data_path).evaluate(loaded)


def test_mnist_rejects_legacy_test_cache(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist.pt"
    inputs = torch.zeros((10_000, 784), dtype=torch.uint8)
    torch.save((inputs, torch.zeros(10_000, dtype=torch.int64)), data_path)
    loaded = _loaded_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="exactly 70000 samples, found 10000"):
        MnistBenchmark(data_path=data_path).evaluate(loaded)


def test_image_benchmarks_expect_full_dataset() -> None:
    assert MnistBenchmark().expected_samples == 70_000
    assert FashionMnistBenchmark().expected_samples == 70_000
    assert MnistBenchmark(expected_samples=None).expected_samples == 70_000
    assert FashionMnistBenchmark(expected_samples=None).expected_samples == 70_000


def test_mnist_requires_data_file(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    with pytest.raises(FileNotFoundError, match="MNIST dataset is unavailable"):
        MnistBenchmark(data_path=tmp_path / "missing.pt").evaluate(loaded)


def test_mnist_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 10,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 10)), "bias": torch.zeros(10)},
    )
    with pytest.raises(ValueError, match="input_size 784"):
        MnistBenchmark().load_submission(submission)


def test_fashion_mnist_evaluation_and_threshold(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = FashionMnistBenchmark(
        data_path=data_path, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}


def test_fashion_mnist_normalisation_flag(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    _write_mnist_cache(data_path)
    loaded = _loaded_fashion_mnist_submission(tmp_path, normalisation=True)
    score, score_std, passed, details = FashionMnistBenchmark(
        data_path=data_path, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": True}


def test_fashion_mnist_accepts_two_dimensional_labels(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    inputs[0, 0] = 255
    inputs[1, 1] = 255
    _write_mnist_cache_with(data_path, inputs, torch.tensor([[0], [1]]))
    loaded = _loaded_fashion_mnist_submission(tmp_path)
    score, _, passed, details = FashionMnistBenchmark(
        data_path=data_path, expected_samples=2
    ).evaluate(loaded)
    assert score == 1.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}


def test_fashion_mnist_rejects_length_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    _write_mnist_cache_with(
        data_path, torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1, 2])
    )
    loaded = _loaded_fashion_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="must contain raw"):
        FashionMnistBenchmark(data_path=data_path, expected_samples=2).evaluate(loaded)


def test_fashion_mnist_rejects_wrong_input_width(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    _write_mnist_cache_with(
        data_path, torch.zeros((2, 10), dtype=torch.uint8), torch.tensor([0, 1])
    )
    loaded = _loaded_fashion_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="must contain raw"):
        FashionMnistBenchmark(data_path=data_path, expected_samples=2).evaluate(loaded)


def test_fashion_mnist_rejects_partial_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    _write_mnist_cache(data_path)
    loaded = _loaded_fashion_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="exactly 70000 samples, found 2"):
        FashionMnistBenchmark(data_path=data_path).evaluate(loaded)


def test_fashion_mnist_rejects_legacy_test_cache(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist.pt"
    inputs = torch.zeros((10_000, 784), dtype=torch.uint8)
    torch.save((inputs, torch.zeros(10_000, dtype=torch.int64)), data_path)
    loaded = _loaded_fashion_mnist_submission(tmp_path)
    with pytest.raises(SubmissionError, match="exactly 70000 samples, found 10000"):
        FashionMnistBenchmark(data_path=data_path).evaluate(loaded)


def test_fashion_mnist_requires_data_file(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    with pytest.raises(FileNotFoundError, match="Fashion-MNIST dataset is unavailable"):
        FashionMnistBenchmark(data_path=tmp_path / "missing.pt").evaluate(loaded)


def test_fashion_mnist_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 10,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 10)), "bias": torch.zeros(10)},
    )
    with pytest.raises(ValueError, match="Fashion-MNIST submissions"):
        FashionMnistBenchmark().load_submission(submission)
