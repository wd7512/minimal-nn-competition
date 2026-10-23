import json
from pathlib import Path
from typing import Any

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake import benchmark as snake_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeSimple16x16Benchmark
from minimal_nn_competition.benchmarks.snake.replay import record_game, replay_stem
from minimal_nn_competition.leaderboard.render import (
    _score_text,
    render_site,
    render_submission_pages,
)
from minimal_nn_competition.model.entry import (
    Submission,
    SubmissionConfig,
    materialize_submission,
)
from minimal_nn_competition.runner.entry import BenchmarkResult


def _write_submission(
    tmp_path: Path, config: dict[str, Any], state: dict[str, torch.Tensor]
) -> Submission:
    directory = tmp_path / "example"
    directory.mkdir()
    config = {"name": "example", "author": "tester", **config}
    (directory / "config.json").write_text(json.dumps(config))
    torch.save(state, directory / "model.pt")
    return Submission.from_directory(directory)


def _right_model() -> object:
    class Right:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    return Right()


def test_registry_and_list_include_16x16(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark = create_benchmark("snake-simple-16x16")
    assert benchmark.name == "snake-simple-16x16"
    assert benchmark.size == 16
    assert main(["list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "chess-eval",
        "fashion-mnist",
        "mnist",
        "snake-obstacles-8x8",
        "snake-simple-16x16",
        "snake-simple-8x8",
    ]


def test_config_allows_input_indices_for_16x16(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": "snake-simple-16x16",
                "name": "x",
                "author": "tester",
                "architecture": "mlp",
                "input_size": 3,
                "output_size": 4,
                "hidden_sizes": [4],
                "input_indices": [0, 1, 2],
            }
        )
    )
    assert SubmissionConfig.from_file(path).input_indices == (0, 1, 2)


def test_16x16_evaluation_uses_256_cell_board(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-16x16",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeSimple16x16Benchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert score == 1.0
    assert score_std == 0.0
    assert passed is False
    assert details["seed_lengths"] == [1, 1]
    assert details["seed_moves"] == [9, 9]
    assert details["max_length"] == 256


def test_16x16_full_board_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FullBoardEnvironment:
        size = 16

        def __init__(
            self,
            size: int,
            seed: int,
            max_moves: int,
            input_indices: object = None,
            blocked: object = (),
            loop_on_full: bool = False,
        ) -> None:
            self.snake = [(0, 0)] * 256

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", FullBoardEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-16x16",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeSimple16x16Benchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert score == 256.0
    assert score_std == 0.0
    assert passed is True
    assert details["seed_lengths"] == [256, 256]
    assert details["max_length"] == 256


def test_record_game_records_board_size() -> None:
    first = record_game(_right_model(), seed=0, size=16)
    second = record_game(_right_model(), seed=0, size=16)
    assert first == second
    assert first["size"] == 16
    assert first["frames"][0]["snake"] == [[8, 8]]
    assert first["moves"] == len(first["frames"]) - 1


def test_render_16x16_section_percentage_and_replay(tmp_path: Path) -> None:
    result = BenchmarkResult(
        "example",
        "snake-simple-16x16",
        128.0,
        False,
        20,
        {"max_length": 256, "seed_moves": [10, 20]},
    )
    payload = record_game(_right_model(), size=16)
    index = render_site([result], tmp_path, {"snake-simple-16x16--example": payload})
    html = index.read_text()
    assert "Simple 16×16" in html
    assert "50.00% ± 0.00%" in html
    assert "Game length</th>" not in html
    assert "Replay</th>" not in html
    assert "submissions/snake-simple-16x16--example.html" in html
    game_page = (tmp_path / "submissions" / "snake-simple-16x16--example.html").read_text()
    assert "Simple 16×16" in game_page
    assert "Replay" in game_page
    assert "15.0 ± 5.0" in game_page
    assert 'width="640" height="640"' in game_page


def test_collect_records_16x16_replay(tmp_path: Path, fast_snake_cli: None) -> None:
    directory = tmp_path / "models" / "snake-simple-16x16" / "tiny"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "snake-simple-16x16",
                "name": "tiny",
                "author": "tester",
                "architecture": "linear",
                "input_size": 24,
                "output_size": 4,
            }
        )
    )
    torch.save({"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)}, directory / "model.pt")
    replays = tmp_path / "replays"
    assert (
        main(
            [
                "collect",
                "--models",
                str(tmp_path / "models"),
                "--output",
                str(tmp_path / "leaderboard.json"),
                "--replay-dir",
                str(replays),
            ]
        )
        == 0
    )
    payload = json.loads((replays / "snake-simple-16x16--tiny.json").read_text())
    assert payload["size"] == 16
    assert payload["frames"][0]["snake"] == [[8, 8]]


def test_train_16x16_writes_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_nn_competition.examples import snake_submission

    monkeypatch.setattr(snake_submission, "_evaluate", lambda *args: [1])
    output = tmp_path / "submission"
    snake_submission.train(
        output,
        iterations=1,
        episodes_per_iteration=1,
        max_moves=8,
        update_epochs=1,
        device_name="cpu",
        benchmark="snake-simple-16x16",
    )
    config = json.loads((output / "config.json").read_text())
    assert config["benchmark"] == "snake-simple-16x16"


def test_train_16x16_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_nn_competition.examples import snake_submission

    calls = []

    def fake_train(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(snake_submission, "train", fake_train)
    assert main(["train-snake-simple-16x16", "--output", str(tmp_path / "output")]) == 0
    assert calls[0][1]["benchmark"] == "snake-simple-16x16"


def test_16x16_defaults_are_100_seeds_and_4096_moves() -> None:
    benchmark = SnakeSimple16x16Benchmark()
    assert benchmark.seeds == tuple(range(100))
    assert benchmark.max_moves == 65536
    assert benchmark.moves_per_food == 2 * 16 * 16
    assert benchmark.size == 16


def test_near_perfect_failing_image_score_keeps_precision() -> None:
    result = BenchmarkResult("example", "mnist", 0.99996, False, 10, {})
    assert _score_text(result) == "99.9960%"


def test_replay_links_are_scoped_by_benchmark(tmp_path: Path) -> None:
    payload_8 = record_game(_right_model())
    payload_16 = record_game(_right_model(), size=16)
    results = [
        BenchmarkResult("example", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64}),
        BenchmarkResult("example", "snake-simple-16x16", 128.0, False, 20, {"max_length": 256}),
    ]
    links = render_submission_pages(
        results,
        {
            replay_stem("snake-simple-8x8", "example"): payload_8,
            replay_stem("snake-simple-16x16", "example"): payload_16,
        },
        tmp_path,
    )
    assert links[("snake-simple-8x8", "example")] == "submissions/snake-simple-8x8--example.html"
    assert links[("snake-simple-16x16", "example")] == (
        "submissions/snake-simple-16x16--example.html"
    )
    assert (tmp_path / links[("snake-simple-8x8", "example")]).is_file()
    assert (tmp_path / links[("snake-simple-16x16", "example")]).is_file()


def test_16x16_rejects_data_path() -> None:
    assert (
        main(
            [
                "evaluate",
                "snake-simple-16x16",
                "models/snake-simple-16x16/example",
                "--data-path",
                "data.pt",
            ]
        )
        == 2
    )


def test_16x16_rejects_out_of_range_input_indices(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-16x16",
            "architecture": "linear",
            "input_size": 1,
            "output_size": 4,
            "input_indices": [24],
        },
        {"weight": torch.zeros((4, 1)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_indices"):
        SnakeSimple16x16Benchmark().load_submission(submission)


def test_train_16x16_scales_the_default_move_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minimal_nn_competition.benchmarks.snake import environment as snake_environment
    from minimal_nn_competition.examples import snake_submission

    captured = []

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.snake = [(8, 8)]
            self.food = (8, 9)

        def reset(self) -> tuple[float, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
            return (0.0,) * 24, 0.0, True

    monkeypatch.setattr(snake_environment, "SnakeEnvironment", FakeEnvironment)
    monkeypatch.setattr(snake_submission, "_evaluate", lambda *args: [1])
    snake_submission.train(
        tmp_path / "submission",
        iterations=1,
        episodes_per_iteration=1,
        update_epochs=1,
        device_name="cpu",
        benchmark="snake-simple-16x16",
    )
    assert captured[0]["size"] == 16
    assert captured[0]["max_moves"] == 16 * 16 * 4
