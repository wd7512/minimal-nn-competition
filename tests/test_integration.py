import json
from pathlib import Path

import pytest
import torch

from minimal_nn_competition import __version__
from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.chess_eval.benchmark import (
    ChessEval10kBenchmark,
    ChessEval100kBenchmark,
)
from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import FashionMnistBenchmark
from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark
from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.runner.entry import BenchmarkResult


def test_checked_in_examples_load() -> None:
    chess = Submission.from_directory(Path("models/chess-eval-10k/example"))
    chess_100k = Submission.from_directory(Path("models/chess-eval-100k/example"))
    fashion_mnist = Submission.from_directory(Path("models/fashion-mnist/example"))
    mnist = Submission.from_directory(Path("models/mnist/example"))
    snake = Submission.from_directory(Path("models/snake-simple-8x8/example"))
    assert ChessEval10kBenchmark().load_submission(chess).model is not None
    assert ChessEval100kBenchmark().load_submission(chess_100k).model is not None
    assert FashionMnistBenchmark().load_submission(fashion_mnist).model is not None
    assert MnistBenchmark().load_submission(mnist).model is not None
    assert SnakeBenchmark().load_submission(snake).model is not None


def test_evaluate_snake_cli(capsys: pytest.CaptureFixture[str], fast_snake_cli: None) -> None:
    assert main(["evaluate", "snake-simple-8x8", "models/snake-simple-8x8/example"]) == 0
    out = capsys.readouterr().out
    assert '"benchmark": "snake-simple-8x8"' in out
    assert '"name": "example"' in out
    assert '"author": "example author"' in out


def test_evaluate_chess_eval_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "chess-eval-10k", "models/chess-eval-10k/example"]) == 0
    out = capsys.readouterr().out
    assert '"benchmark": "chess-eval-10k"' in out
    assert '"name": "chess-mlp-64"' in out
    assert '"author": "example author"' in out
    assert '"passed": true' in out


def test_evaluate_fashion_mnist_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FashionMnistBenchmark, "expected_samples", 2)
    data_path = tmp_path / "fashion-mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    torch.save((inputs, torch.tensor([0, 1])), data_path)
    assert (
        main(
            [
                "evaluate",
                "fashion-mnist",
                "models/fashion-mnist/example",
                "--data-path",
                str(data_path),
            ]
        )
        == 0
    )
    assert '"benchmark": "fashion-mnist"' in capsys.readouterr().out


def test_evaluate_mnist_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MnistBenchmark, "expected_samples", 2)
    data_path = tmp_path / "mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    torch.save((inputs, torch.tensor([0, 1])), data_path)
    assert (
        main(
            [
                "evaluate",
                "mnist",
                "models/mnist/example",
                "--data-path",
                str(data_path),
            ]
        )
        == 0
    )
    assert '"benchmark": "mnist"' in capsys.readouterr().out


def test_evaluate_mnist_uses_default_data_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MnistBenchmark, "expected_samples", 2)
    submission = Path("models/mnist/example").resolve()
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(
        (torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1])),
        data_dir / "mnist.pt",
    )
    assert main(["evaluate", "mnist", str(submission)]) == 0
    assert '"benchmark": "mnist"' in capsys.readouterr().out


def test_evaluate_fashion_mnist_uses_default_data_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FashionMnistBenchmark, "expected_samples", 2)
    submission = Path("models/fashion-mnist/example").resolve()
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(
        (torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1])),
        data_dir / "fashion-mnist.pt",
    )
    assert main(["evaluate", "fashion-mnist", str(submission)]) == 0
    assert '"benchmark": "fashion-mnist"' in capsys.readouterr().out


def test_data_path_rejected_for_snake(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "evaluate",
                "snake-simple-8x8",
                "models/snake-simple-8x8/example",
                "--data-path",
                "data.pt",
            ]
        )
        == 2
    )
    assert capsys.readouterr().out == ""


def test_render_shows_name_and_author(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "example",
            "snake-simple-8x8",
            8.0,
            False,
            20,
            {"max_length": 64},
            name="Cool Net",
            author="tester",
        ),
    ]
    html = render_site(results, tmp_path).read_text()
    assert "Cool Net" in html
    assert 'href="submissions/snake-simple-8x8--example.html"' in html
    detail = (tmp_path / "submissions" / "snake-simple-8x8--example.html").read_text()
    assert "<dt>Name</dt>" in detail
    assert "<dt>Author</dt>" in detail
    assert "Cool Net" in detail
    assert "tester" in detail


def test_render_site(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "fashion-example",
            "fashion-mnist",
            0.75,
            False,
            10,
            {"samples": 2, "normalisation": True},
        ),
        BenchmarkResult(
            "mnist-example", "mnist", 0.5, False, 10, {"samples": 2, "normalisation": False}
        ),
        BenchmarkResult(
            "snake-example",
            "snake-simple-8x8",
            1.0,
            False,
            20,
            {"seed_lengths": [1], "max_length": 64, "seed_moves": [10, 20]},
            score_std=0.5,
        ),
    ]
    index = render_site(results, tmp_path)
    html = index.read_text()
    assert 'id="snake"' in html
    assert 'id="mnist"' in html
    assert "Fashion-MNIST" in html
    assert "Simple 8×8" in html
    assert "fashion-example" in html
    assert "mnist-example" in html
    assert 'href="internal/speed/"' in html
    assert f"v{__version__}" in html
    assert "50.00%" in html
    assert "1.56%" in html
    assert "1.56% ±" not in html
    assert "Game length</th>" in html
    assert "Samples</th>" not in html
    assert "Normalisation</th>" not in html
    assert "submissions/snake-simple-8x8--snake-example.html" in html
    assert "submissions/mnist--mnist-example.html" in html
    snake_page = (tmp_path / "submissions" / "snake-simple-8x8--snake-example.html").read_text()
    assert "<dt>Game length</dt>" in snake_page
    assert "15.0 ± 5.0" in snake_page
    mnist_page = (tmp_path / "submissions" / "mnist--mnist-example.html").read_text()
    assert "<dt>Normalisation</dt>" in mnist_page
    assert "<dd>false</dd>" in mnist_page
    assert "<th>Details</th>" not in html
    assert "<details>" not in html
    raw = json.loads((tmp_path / "leaderboard.json").read_text())
    assert [item["submission"] for item in raw] == [
        "fashion-example",
        "mnist-example",
        "snake-example",
    ]
    assert raw[2]["score_std"] == 0.5


def test_render_near_perfect_failure_is_not_shown_as_100(tmp_path: Path) -> None:
    near_miss = BenchmarkResult(
        "mnist-near",
        "mnist",
        69_999 / 70_000,
        False,
        10,
        {"samples": 70_000, "normalisation": False},
    )
    html = render_site([near_miss], tmp_path / "near").read_text()
    assert "100.00%" not in html
    assert "99.9986%" in html

    perfect = BenchmarkResult(
        "mnist-perfect",
        "mnist",
        1.0,
        True,
        10,
        {"samples": 70_000, "normalisation": False},
    )
    html = render_site([perfect], tmp_path / "perfect").read_text()
    assert "100.00%" in html


def test_render_mnist_has_no_game_length_column(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "mnist-example", "mnist", 0.5, False, 10, {"samples": 2, "normalisation": False}
        ),
        BenchmarkResult(
            "fashion-example",
            "fashion-mnist",
            0.5,
            False,
            10,
            {"samples": 2, "normalisation": True},
        ),
    ]
    html = render_site(results, tmp_path).read_text()
    assert "Game length</th>" not in html
    assert "Samples</th>" not in html
    mnist_page = (tmp_path / "submissions" / "mnist--mnist-example.html").read_text()
    assert "<dt>Samples</dt>" not in mnist_page


def test_render_chess_eval(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "chess-example",
            "chess-eval-10k",
            1.0,
            True,
            53_505,
            {"tolerance_pawns": 0.1, "mae": 0.0046, "samples": 10_000},
            name="chess-mlp-64",
            author="tester",
        )
    ]
    html = render_site(results, tmp_path).read_text()
    assert "Chess evaluation" in html
    assert "Stockfish 10k" in html
    assert "100.00%" in html
    assert "100.00% ±" not in html
    assert "Error</th>" in html
    assert "0.0046" in html
    assert "Samples</th>" not in html
    assert "mean error" in html
    assert "chess-example" in html
    detail = (tmp_path / "submissions" / "chess-eval-10k--chess-example.html").read_text()
    assert "<dt>Mean error</dt>" in detail
    assert "0.0046 pawns" in detail
    assert "<dt>Samples</dt>" not in detail


def test_rank_chess_eval_breaks_score_ties_by_mean_error() -> None:
    def chess_result(submission: str, mae: float) -> BenchmarkResult:
        return BenchmarkResult(
            submission,
            "chess-eval-10k",
            1.0,
            True,
            10,
            {"tolerance_pawns": 0.1, "mae": mae, "samples": 10_000},
        )

    ranked = rank_results([chess_result("worse", 0.02), chess_result("better", 0.01)])
    assert [result.submission for result in ranked] == ["better", "worse"]


def test_render_snake_caption_explains_failed_game_tiebreak(tmp_path: Path) -> None:
    results = [
        BenchmarkResult("obstacle-a", "snake-obstacles-8x8", 5.1, False, 50, {"seed_moves": [30]}),
        BenchmarkResult("obstacle-b", "snake-obstacles-8x8", 5.1, False, 200, {"seed_moves": [90]}),
    ]
    html = render_site(results, tmp_path).read_text()
    assert "failed submissions rank by longest game first" in html
    # Longest game ranks first despite having more parameters.
    assert html.index("obstacle-b") < html.index("obstacle-a")


def test_render_groups_versions_and_defaults(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            "snake-8", "snake-simple-8x8", 1.0, False, 10, {"seed_lengths": [1], "max_length": 64}
        ),
        BenchmarkResult("snake-16", "snake-simple-16x16", 0.5, False, 10, {"max_length": 256}),
        BenchmarkResult("mnist", "mnist", 0.5, False, 10, {"normalisation": False}),
        BenchmarkResult("fashion", "fashion-mnist", 0.5, False, 10, {"normalisation": True}),
        BenchmarkResult("chess-10k", "chess-eval-10k", 1.0, True, 10, {"tolerance_pawns": 0.1}),
        BenchmarkResult(
            "chess-100k",
            "chess-eval-100k",
            0.5,
            False,
            10,
            {"tolerance_pawns": 0.1, "samples": 100_000},
        ),
    ]
    html = render_site(results, tmp_path).read_text()
    assert html.index('id="snake"') < html.index('id="mnist"') < html.index('id="chess"')
    assert 'data-version="snake-simple-8x8"' in html
    assert 'data-version="snake-simple-16x16"' in html
    assert 'class="version-panel is-active" id="panel-snake-simple-8x8"' in html
    assert 'class="version-panel" id="panel-snake-simple-16x16"' in html
    assert "data-version-button" in html
    assert 'aria-controls="panel-fashion-mnist"' in html
    chess_start = html.index('id="chess"')
    chess_end = html.index("</section>", chess_start)
    chess_html = html[chess_start:chess_end]
    assert 'data-version="chess-eval-10k"' in chess_html
    assert 'data-version="chess-eval-100k"' in chess_html
    assert 'class="version-panel is-active" id="panel-chess-eval-10k"' in chess_html
    assert 'class="version-panel" id="panel-chess-eval-100k"' in chess_html
    # Regression: the tab script must not rewrite panel DOM. Group-level
    # run/pass stats were removed, so no [data-count]/[data-pass] hooks may
    # remain for querySelector to resolve to a panel wrapper.
    assert "data-count" not in html
    assert 'querySelector("[data-pass]")' not in html


def test_render_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from minimal_nn_competition.leaderboard import save_results

    results_path = tmp_path / "leaderboard.json"
    save_results(
        results_path,
        [
            BenchmarkResult(
                "snake-example", "snake-simple-8x8", 1.0, False, 20, {"seed_lengths": [1]}
            )
        ],
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
    shutil.copytree(
        Path("models/snake-simple-8x8/example"), models / "snake-simple-8x8" / "example"
    )
    output = tmp_path / "leaderboard.json"
    assert main(["collect", "--models", str(models), "--output", str(output)]) == 0
    results = json.loads(output.read_text())
    assert results[0]["benchmark"] == "snake-simple-8x8"
    assert results[0]["submission"] == "example"
    assert capsys.readouterr().out.strip() == str(output)


def test_collect_fashion_mnist_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    monkeypatch.setattr(FashionMnistBenchmark, "expected_samples", 2)
    models = tmp_path / "models"
    shutil.copytree(Path("models/fashion-mnist/example"), models / "fashion-mnist" / "example")
    data_path = tmp_path / "fashion-mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    torch.save((inputs, torch.tensor([0, 1])), data_path)
    output = tmp_path / "leaderboard.json"
    assert (
        main(
            [
                "collect",
                "--models",
                str(models),
                "--output",
                str(output),
                "--fashion-mnist-data-path",
                str(data_path),
            ]
        )
        == 0
    )
    results = json.loads(output.read_text())
    assert results[0]["benchmark"] == "fashion-mnist"
    assert results[0]["submission"] == "example"
    assert capsys.readouterr().out.strip() == str(output)


def test_collect_mnist_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    monkeypatch.setattr(MnistBenchmark, "expected_samples", 2)
    models = tmp_path / "models"
    shutil.copytree(Path("models/mnist/example"), models / "mnist" / "example")
    data_path = tmp_path / "mnist.pt"
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    torch.save((inputs, torch.tensor([0, 1])), data_path)
    output = tmp_path / "leaderboard.json"
    assert (
        main(
            [
                "collect",
                "--models",
                str(models),
                "--output",
                str(output),
                "--data-path",
                str(data_path),
            ]
        )
        == 0
    )
    results = json.loads(output.read_text())
    assert results[0]["benchmark"] == "mnist"
    assert results[0]["submission"] == "example"
    assert capsys.readouterr().out.strip() == str(output)


def test_collect_rejects_partial_mnist_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    submission = Path("models/mnist/example").resolve()
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    shutil.copytree(submission, models / "mnist" / "example")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(
        (torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1])),
        data_dir / "mnist.pt",
    )
    output = tmp_path / "leaderboard.json"

    assert main(["collect", "--models", str(models), "--output", str(output)]) == 2
    assert not output.exists()


def test_collect_rejects_partial_fashion_mnist_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    submission = Path("models/fashion-mnist/example").resolve()
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    shutil.copytree(submission, models / "fashion-mnist" / "example")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(
        (torch.zeros((2, 784), dtype=torch.uint8), torch.tensor([0, 1])),
        data_dir / "fashion-mnist.pt",
    )
    output = tmp_path / "leaderboard.json"

    assert main(["collect", "--models", str(models), "--output", str(output)]) == 2
    assert not output.exists()
