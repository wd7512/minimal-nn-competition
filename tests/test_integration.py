import json
from pathlib import Path

import pytest

from minimal_nn_competition import __version__
from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.runner.entry import BenchmarkResult


def test_checked_in_examples_load() -> None:
    mnist = Submission.from_directory(Path("models/mnist/example"))
    snake = Submission.from_directory(Path("models/snake/example"))
    assert MnistBenchmark().load_submission(mnist).model is not None
    assert SnakeBenchmark().load_submission(snake).model is not None


def test_evaluate_snake_cli(capsys: pytest.CaptureFixture[str], fast_snake_cli: None) -> None:
    assert main(["evaluate", "snake", "models/snake/example"]) == 0
    out = capsys.readouterr().out
    assert '"benchmark": "snake"' in out
    assert '"name": "example"' in out
    assert '"author": "example author"' in out


def test_data_path_rejected_for_snake(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "snake", "models/snake/example", "--data-path", "data.pt"]) == 2
    assert capsys.readouterr().out == ""


def test_render_shows_name_and_author(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "example",
            "snake",
            8.0,
            False,
            20,
            {"max_length": 64},
            name="Cool Net",
            author="tester",
        ),
    ]
    html = render_site(results, tmp_path).read_text()
    assert "<th>Name</th>" in html
    assert "<th>Author</th>" in html
    assert "Cool Net" in html
    assert "tester" in html


def test_render_site(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "mnist-example", "mnist", 0.5, False, 10, {"samples": 2, "normalisation": False}
        ),
        BenchmarkResult(
            "snake-example",
            "snake",
            1.0,
            False,
            20,
            {"seed_lengths": [1], "max_length": 64, "seed_moves": [10, 20]},
            score_std=0.5,
        ),
    ]
    index = render_site(results, tmp_path)
    html = index.read_text()
    assert "MNIST COMPRESSION" in html
    assert "SNAKE GAME" in html
    assert "mnist-example" in html
    assert 'href="internal/speed/"' in html
    assert f"v{__version__}" in html
    assert "50.00%" in html
    assert "1.56% ± 0.78%" in html
    assert "<th>Game length</th>" in html
    assert "15.0 ± 5.0" in html
    assert "<th>Normalisation</th>" in html
    assert "<td>false</td>" in html
    assert "<th>Details</th>" not in html
    assert "<details>" not in html
    raw = json.loads((tmp_path / "leaderboard.json").read_text())
    assert [item["submission"] for item in raw] == ["mnist-example", "snake-example"]
    assert raw[1]["score_std"] == 0.5


def test_render_mnist_has_no_game_length_column(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "mnist-example", "mnist", 0.5, False, 10, {"samples": 2, "normalisation": False}
        )
    ]
    html = render_site(results, tmp_path).read_text()
    assert "<th>Game length</th>" not in html


def test_render_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from minimal_nn_competition.leaderboard import save_results

    results_path = tmp_path / "leaderboard.json"
    save_results(
        results_path,
        [BenchmarkResult("snake-example", "snake", 1.0, False, 20, {"seed_lengths": [1]})],
    )
    output = tmp_path / "docs"
    assert main(["render", str(results_path), str(output)]) == 0
    assert (output / "index.html").is_file()
    assert capsys.readouterr().out.strip().endswith("index.html")


def test_collect_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fast_snake_cli: None
) -> None:
    import shutil

    models = tmp_path / "models"
    shutil.copytree(Path("models/snake/example"), models / "snake" / "example")
    output = tmp_path / "leaderboard.json"
    assert main(["collect", "--models", str(models), "--output", str(output)]) == 0
    results = json.loads(output.read_text())
    assert results[0]["benchmark"] == "snake"
    assert results[0]["submission"] == "example"
    assert capsys.readouterr().out.strip() == str(output)
