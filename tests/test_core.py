from pathlib import Path

import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.leaderboard import load_results, rank_results, save_results
from minimal_nn_competition.model.entry import SubmissionConfig, SubmissionError, parameter_count
from minimal_nn_competition.runner.entry import BenchmarkResult


class FakeTensor:
    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


def test_parameter_count_rejects_non_tensors() -> None:
    with pytest.raises(SubmissionError, match="not a tensor"):
        parameter_count({"weight": object()})


def test_parameter_count_sums_tensor_values() -> None:
    assert parameter_count({"a": FakeTensor(3), "b": FakeTensor(5)}) == 8


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake","architecture":"mlp","input_size":24,'
        '"output_size":4,"hidden_sizes":[8,4],"activation":"relu",'
        '"normalisation":true}'
    )
    config = SubmissionConfig.from_file(path)
    assert config.benchmark == "snake"
    assert config.hidden_sizes == (8, 4)
    assert config.normalisation is True


def test_config_accepts_input_indices(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake","architecture":"linear","input_size":5,'
        '"output_size":4,"input_indices":[0,1,2,4,7]}'
    )
    config = SubmissionConfig.from_file(path)
    assert config.input_indices == (0, 1, 2, 4, 7)


def test_config_rejects_duplicate_input_indices(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake","architecture":"linear","input_size":2,'
        '"output_size":4,"input_indices":[0,0]}'
    )
    with pytest.raises(SubmissionError, match="duplicates"):
        SubmissionConfig.from_file(path)


def test_config_rejects_linear_hidden_sizes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake","architecture":"linear","input_size":24,'
        '"output_size":4,"hidden_sizes":[8]}'
    )
    with pytest.raises(SubmissionError, match="must not declare hidden_sizes"):
        SubmissionConfig.from_file(path)


def test_snake_is_deterministic() -> None:
    first = SnakeEnvironment(seed=4)
    second = SnakeEnvironment(seed=4)
    assert first.reset() == second.reset()
    assert first.step(1) == second.step(1)


def test_snake_can_move_into_vacating_tail() -> None:
    environment = SnakeEnvironment(seed=4)
    environment.reset()
    environment.snake = [(4, 4), (4, 5), (3, 5), (3, 4)]
    environment.direction = (-1, 0)
    environment.food = (0, 0)
    _, _, done = environment.step(3)
    assert done is False
    assert environment.snake[0] == (3, 4)


def test_registry_and_leaderboard_ranking() -> None:
    assert create_benchmark("snake").name == "snake"
    results = [
        BenchmarkResult("a", "snake", 4, False, 1, {}),
        BenchmarkResult("b", "snake", 2, True, 20, {}),
        BenchmarkResult("c", "snake", 2, True, 10, {}),
        BenchmarkResult("d", "mnist", 0.9, False, 5, {}),
    ]
    assert [result.submission for result in rank_results(results)] == ["d", "c", "b", "a"]


def test_leaderboard_round_trip(tmp_path: Path) -> None:
    result = BenchmarkResult("example", "snake", 1, True, 8, {"seed_lengths": [1]}, score_std=0.5)
    path = tmp_path / "leaderboard.json"
    save_results(path, [result])
    assert load_results(path) == [result]


def test_leaderboard_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "leaderboard.json"
    save_results(path, [BenchmarkResult("example", "snake", 1, True, 8, {})])
    assert main(["leaderboard", str(path)]) == 0
    assert '"benchmark": "snake"' in capsys.readouterr().out
