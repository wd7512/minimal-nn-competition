import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.benchmarks.registry import WEIGHT_BENCHMARKS, create_benchmark
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions import artifacts
from minimal_nn_competition.submissions.artifacts import _load_replays, _record_replay


def _submission(tmp_path: Path, benchmark: str, hidden: list[int], *, normalise=True) -> Submission:
    directory = tmp_path / "models" / benchmark / "tiny"
    directory.mkdir(parents=True)
    sizes = [784, *hidden, 10]
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
                "input_size": 784,
                "output_size": 10,
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
    submission = _submission(tmp_path, benchmark, [8, 6])
    games = tmp_path / "docs" / "games"
    _record_replay(games, "tiny", create_benchmark(benchmark), submission)
    path = games / f"{benchmark}--tiny.json"
    payload = json.loads(path.read_text())
    assert (len(payload["labels"] or []) == 10) == (benchmark == "fashion-mnist")
    pixels = [float(i % 256) for i in range(784)]
    model = create_benchmark(benchmark).load_submission(submission).model
    expected = model(torch.tensor(pixels)[None] / 255)[0]
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
    assert 'id="pad"' in page and '"layers"' in page
    assert ("look like noise" in page) == benchmark.endswith("-shuffled")


def test_oversized_models_are_not_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_WEIGHT_PARAMETERS", 10)
    submission = _submission(tmp_path, "mnist", [8])
    games = tmp_path / "games"
    _record_replay(games, "tiny", create_benchmark("mnist"), submission)
    assert not list(games.glob("*"))
