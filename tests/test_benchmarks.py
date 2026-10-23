import json
from pathlib import Path
from typing import Any

import pytest
import torch

from minimal_nn_competition.benchmarks.connect4.benchmark import Connect4Benchmark
from minimal_nn_competition.benchmarks.connect4.engine import (
    legal_moves,
    opponent_move,
    winner,
)
from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import FashionMnistBenchmark
from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake import benchmark as snake_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark
from minimal_nn_competition.model.entry import Submission, materialize_submission


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
            "benchmark": "snake",
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

        def __init__(self, seed: int, max_moves: int, input_indices: object = None) -> None:
            self.snake = [(0, 0)] * 64

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", FullBoardEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
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

        def __init__(self, seed: int, max_moves: int, input_indices: object = None) -> None:
            self.snake = [(0, 0)] * (64 if seed == 0 else 1)

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", VariableEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
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
            "benchmark": "snake",
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
            "benchmark": "snake",
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
            "benchmark": "snake",
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
            "benchmark": "snake",
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
    data_path = tmp_path / "mnist-test.pt"
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
    score, score_std, passed, details = MnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}

    score, _, passed, _ = MnistBenchmark(data_path=data_path, threshold=1.01).evaluate(loaded)
    assert score == 1.0
    assert passed is False


def test_mnist_normalisation_flag(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist-test.pt"
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
    score, score_std, passed, details = MnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": True}


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
    with pytest.raises(FileNotFoundError, match="MNIST test data is unavailable"):
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
    data_path = tmp_path / "fashion-mnist-test.pt"
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
    score, score_std, passed, details = FashionMnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}


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
    with pytest.raises(FileNotFoundError, match="Fashion-MNIST test data is unavailable"):
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


def _connect4_board(cells: dict[tuple[int, int], int]) -> tuple[int, ...]:
    board = [0] * 42
    for (row, col), player in cells.items():
        board[row * 7 + col] = player
    return tuple(board)


def test_connect4_registers_by_name() -> None:
    assert create_benchmark("connect4").name == "connect4"
    assert Connect4Benchmark().seeds == tuple(range(20))


def test_connect4_win_detection() -> None:
    horizontal = _connect4_board({(5, 0): 1, (5, 1): 1, (5, 2): 1, (5, 3): 1})
    assert winner(horizontal) == 1
    vertical = _connect4_board({(5, 0): -1, (4, 0): -1, (3, 0): -1, (2, 0): -1})
    assert winner(vertical) == -1
    diagonal = _connect4_board({(2, 0): 1, (3, 1): 1, (4, 2): 1, (5, 3): 1})
    assert winner(diagonal) == 1
    assert winner((0,) * 42) is None
    full_column = _connect4_board({(row, 0): 1 if row % 2 == 0 else -1 for row in range(6)})
    assert 0 not in legal_moves(full_column)
    assert legal_moves((0,) * 42) == list(range(7))


def test_connect4_opponent_blocks_immediate_win() -> None:
    threat = _connect4_board({(5, 0): 1, (5, 1): 1, (5, 2): 1})
    assert opponent_move(threat) == 3
    winning = _connect4_board({(5, 0): -1, (5, 1): -1, (5, 2): -1})
    assert opponent_move(winning) == 3
    assert opponent_move((0,) * 42) == 3
    assert opponent_move((1,) * 42) is None


def test_connect4_evaluate_dummy_linear(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "connect4",
            "architecture": "linear",
            "input_size": 42,
            "output_size": 7,
        },
        {"weight": torch.zeros((7, 42)), "bias": torch.zeros(7)},
    )
    score, score_std, passed, details = Connect4Benchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert isinstance(score, float)
    assert isinstance(score_std, float)
    assert isinstance(passed, bool)
    assert details["seeds"] == 2
    assert len(details["scores"]) == 2
    assert set(details["results"]) <= {"win", "draw", "loss"}


def test_connect4_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "connect4",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 7,
        },
        {"weight": torch.zeros((7, 24)), "bias": torch.zeros(7)},
    )
    with pytest.raises(ValueError, match="input_size 42"):
        Connect4Benchmark().load_submission(submission)
