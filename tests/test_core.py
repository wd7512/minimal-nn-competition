from pathlib import Path

import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.common.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.registry import create_benchmark
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
        '{"benchmark":"snake-simple-8x8","name":"example","author":"tester",'
        '"architecture":"mlp","input_size":24,'
        '"output_size":4,"hidden_sizes":[8,4],"activation":"relu",'
        '"normalisation":true,"description":"A tiny PPO policy."}'
    )
    config = SubmissionConfig.from_file(path)
    assert config.benchmark == "snake-simple-8x8"
    assert config.name == "example"
    assert config.author == "tester"
    assert config.description == "A tiny PPO policy."
    assert config.hidden_sizes == (8, 4)
    assert config.normalisation is True


def test_config_description_is_optional(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","name":"example","author":"tester",'
        '"architecture":"linear","input_size":24,"output_size":4}'
    )
    config = SubmissionConfig.from_file(path)
    assert config.description == ""


def test_config_accepts_input_indices(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","name":"example","author":"tester",'
        '"architecture":"linear","input_size":5,'
        '"output_size":4,"input_indices":[0,1,2,4,7]}'
    )
    config = SubmissionConfig.from_file(path)
    assert config.input_indices == (0, 1, 2, 4, 7)


def test_config_rejects_duplicate_input_indices(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","name":"example","author":"tester",'
        '"architecture":"linear","input_size":2,'
        '"output_size":4,"input_indices":[0,0]}'
    )
    with pytest.raises(SubmissionError, match="duplicates"):
        SubmissionConfig.from_file(path)


def test_config_rejects_missing_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","architecture":"mlp","input_size":24,"output_size":4}'
    )
    with pytest.raises(SubmissionError, match="Invalid submission config"):
        SubmissionConfig.from_file(path)


def test_config_rejects_linear_hidden_sizes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","name":"example","author":"tester",'
        '"architecture":"linear","input_size":24,'
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
    assert create_benchmark("fashion-mnist").name == "fashion-mnist"
    assert create_benchmark("snake-simple-8x8").name == "snake-simple-8x8"
    results = [
        BenchmarkResult("a", "snake-simple-8x8", 4, False, 1, {}),
        BenchmarkResult("b", "snake-simple-8x8", 2, True, 20, {}),
        BenchmarkResult("c", "snake-simple-8x8", 2, True, 10, {}),
        BenchmarkResult("d", "mnist", 0.9, False, 5, {}),
        BenchmarkResult("e", "snake-simple-8x8", 2, True, 10, {"seed_moves": [100, 200]}),
        BenchmarkResult("f", "snake-simple-8x8", 2, True, 10, {"seed_moves": [50, 50]}),
        BenchmarkResult("g", "snake-simple-8x8", 2, True, 9, {"seed_moves": [500]}),
    ]
    assert [result.submission for result in rank_results(results)] == [
        "d",
        "g",
        "f",
        "e",
        "c",
        "b",
        "a",
    ]


def test_rank_snake_failed_ties_favour_longest_game() -> None:
    results = [
        BenchmarkResult("small", "snake-obstacles-8x8", 5.0, False, 10, {"seed_moves": [50]}),
        BenchmarkResult("large", "snake-obstacles-8x8", 5.0, False, 500, {"seed_moves": [200]}),
        BenchmarkResult("unknown", "snake-obstacles-8x8", 5.0, False, 1, {}),
        BenchmarkResult("gone", "snake-obstacles-8x8", 5.0, False, 1, {"seed_moves": []}),
        BenchmarkResult("winner", "snake-obstacles-8x8", 50.0, True, 900, {"seed_moves": [10]}),
    ]
    assert [result.submission for result in rank_results(results)] == [
        "winner",
        "large",
        "small",
        "gone",
        "unknown",
    ]


def test_rank_snake_keeps_score_over_parameters() -> None:
    results = [
        BenchmarkResult("tiny-zero", "snake-simple-8x8", 0, True, 1, {"seed_moves": [999]}),
        BenchmarkResult("any-positive", "snake-simple-8x8", 0.1, True, 900, {"seed_moves": [2]}),
    ]
    assert [result.submission for result in rank_results(results)] == [
        "any-positive",
        "tiny-zero",
    ]


def test_leaderboard_round_trip(tmp_path: Path) -> None:
    result = BenchmarkResult("example", "snake-simple-8x8", 1, True, 8, {"seed_lengths": [1]})
    path = tmp_path / "leaderboard.json"
    save_results(path, [result])
    assert load_results(path) == [result]


def test_leaderboard_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "leaderboard.json"
    save_results(path, [BenchmarkResult("example", "snake-simple-8x8", 1, True, 8, {})])
    assert main(["leaderboard", str(path)]) == 0
    assert '"benchmark": "snake-simple-8x8"' in capsys.readouterr().out
