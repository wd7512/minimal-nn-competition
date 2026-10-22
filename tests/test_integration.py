import json
from pathlib import Path

import pytest

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


def test_evaluate_snake_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "snake", "models/snake/example"]) == 0
    assert '"benchmark": "snake"' in capsys.readouterr().out


def test_data_path_rejected_for_snake(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "snake", "models/snake/example", "--data-path", "data.pt"]) == 2
    assert capsys.readouterr().out == ""


def test_render_site(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "mnist-example", "mnist", 0.5, False, 10, {"samples": 2, "normalisation": False}
        ),
        BenchmarkResult("snake-example", "snake", 1.0, False, 20, {"seed_lengths": [1]}),
    ]
    index = render_site(results, tmp_path)
    html = index.read_text()
    assert "MNIST" in html
    assert "SNAKE" in html
    assert "mnist-example" in html
    assert "<th>Normalisation</th>" in html
    assert "<td>false</td>" in html
    raw = json.loads((tmp_path / "leaderboard.json").read_text())
    assert [item["submission"] for item in raw] == ["mnist-example", "snake-example"]


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


def test_collect_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import shutil

    models = tmp_path / "models"
    shutil.copytree(Path("models/snake/example"), models / "snake" / "example")
    output = tmp_path / "leaderboard.json"
    assert main(["collect", "--models", str(models), "--output", str(output)]) == 0
    results = json.loads(output.read_text())
    assert results[0]["benchmark"] == "snake"
    assert results[0]["submission"] == "example"
    assert capsys.readouterr().out.strip() == str(output)
