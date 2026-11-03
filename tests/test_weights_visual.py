import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.benchmarks.registry import (
    CHESS_BENCHMARKS,
    WEIGHT_BENCHMARKS,
    create_benchmark,
)
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions import artifacts
from minimal_nn_competition.submissions.artifacts import _load_replays, _record_replay


def _submission(
    tmp_path: Path, benchmark: str, hidden: list[int], *, normalise=True, inputs=784, outputs=10
) -> Submission:
    directory = tmp_path / "models" / benchmark / "tiny"
    directory.mkdir(parents=True)
    sizes = [inputs, *hidden, outputs]
    torch.manual_seed(0)
    layers = []
    for a, b in pairwise(sizes):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    model = torch.nn.Sequential(*layers[:-1])
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "name": "tiny",
                "author": "me",
                "architecture": "mlp",
                "input_size": inputs,
                "output_size": outputs,
                "hidden_sizes": hidden,
                "normalisation": normalise,
            }
        )
    )
    return Submission.from_directory(directory)


def _forward(payload: dict, pixels: list[float]) -> list[float]:
    """The same arithmetic as the page's JavaScript."""
    values = [v * payload["scale"] for v in pixels]
    for index, layer in enumerate(payload["layers"]):
        values = [
            sum(w * v for w, v in zip(row, values, strict=True)) + bias
            for row, bias in zip(layer["w"], layer["b"], strict=True)
        ]
        if payload["relu"] and index < len(payload["layers"]) - 1:
            values = [max(0.0, v) for v in values]
    return values


@pytest.mark.parametrize("benchmark", WEIGHT_BENCHMARKS)
def test_exported_weights_match_the_model_and_render_a_widget(tmp_path, benchmark):
    chess = benchmark in CHESS_BENCHMARKS
    size, widget = (773, 'id="chess"') if chess else (784, 'id="pad"')
    submission = _submission(
        tmp_path, benchmark, [8, 6], normalise=not chess, inputs=size, outputs=1 if chess else 10
    )
    games = tmp_path / "docs" / "games"
    _record_replay(games, "tiny", create_benchmark(benchmark), submission)
    path = games / f"{benchmark}--tiny.json"
    payload = json.loads(path.read_text())
    assert (len(payload["labels"] or []) == 10) == (benchmark == "fashion-mnist")
    pixels = [float(i % 2) if chess else float(i % 256) for i in range(size)]
    model = create_benchmark(benchmark).load_submission(submission).model
    expected = model(torch.tensor(pixels)[None] / (1 if chess else 255))[0]
    assert _forward(payload, pixels) == pytest.approx(expected.tolist(), abs=0.05)

    result = BenchmarkResult(
        "tiny",
        benchmark,
        0.5,
        False,
        1,
        {"structure": [784, 8, 6, 10]},
        name="tiny",
        author="me",
        submitted_at="2026-11-03",
    )
    render_site([result], tmp_path / "docs", _load_replays(games))
    page = (tmp_path / "docs" / "submissions" / f"{benchmark}--tiny.html").read_text()
    assert widget in page and '"layers"' in page
    assert ("look like noise" in page) == benchmark.endswith("-shuffled")


def test_oversized_models_are_not_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_WEIGHT_PARAMETERS", 10)
    submission = _submission(tmp_path, "mnist", [8])
    games = tmp_path / "games"
    _record_replay(games, "tiny", create_benchmark("mnist"), submission)
    assert not list(games.glob("*"))


def test_chess_script_encoding_matches_scoring_encoding():
    """Pin the JS board encoding to benchmarks.chess_eval.encode (planes, side, castling)."""
    from minimal_nn_competition.benchmarks import chess_eval
    from minimal_nn_competition.leaderboard.render import _CHESS_SCRIPT

    assert f'P = "{"".join(chess_eval._PIECES)}"' in _CHESS_SCRIPT
    assert "x[768]" in _CHESS_SCRIPT and "x[769 + j]" in _CHESS_SCRIPT
    assert '["K", "Q", "k", "q"]' in _CHESS_SCRIPT
    assert chess_eval.INPUT_SIZE == 773


def test_chess_has_its_own_weight_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_WEIGHT_PARAMETERS", 10)
    submission = _submission(tmp_path, "chess-eval-10k", [8], inputs=773, outputs=1)
    games = tmp_path / "games"
    _record_replay(games, "tiny", create_benchmark("chess-eval-10k"), submission)
    assert (games / "chess-eval-10k--tiny.json").is_file()
