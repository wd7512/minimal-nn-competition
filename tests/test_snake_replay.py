import json
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.snake.replay import (
    REPLAY_SEED,
    load_replay,
    record_game,
    replay_slug,
    save_replay,
)
from minimal_nn_competition.leaderboard.render import (
    _game_length_text,
    render_game_html,
    render_game_pages,
    render_site,
)
from minimal_nn_competition.runner.entry import BenchmarkResult


def _right_model() -> object:
    class Right:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    return Right()


def test_record_game_is_deterministic() -> None:
    first = record_game(_right_model(), seed=REPLAY_SEED)
    second = record_game(_right_model(), seed=REPLAY_SEED)
    assert first == second
    assert first["seed"] == REPLAY_SEED
    assert first["size"] == 8
    assert first["moves"] == len(first["frames"]) - 1
    assert first["frames"][0]["snake"] == [[4, 4]]
    assert first["length"] == len(first["frames"][-1]["snake"])


def test_record_game_requires_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        record_game(_right_model())


def test_replay_slug() -> None:
    assert replay_slug("example") == "example"
    assert replay_slug("my model/v2") == "my_model_v2"
    assert replay_slug("!!!") == "game"


def test_save_and_load_replay_round_trip(tmp_path: Path) -> None:
    payload = record_game(_right_model())
    path = save_replay(tmp_path / "replays" / "example.json", payload)
    assert path.is_file()
    assert load_replay(path) == payload


def test_load_replay_rejects_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="Invalid replay"):
        load_replay(path)


def test_render_game_html() -> None:
    payload = record_game(_right_model())
    page = render_game_html("example", payload)
    assert "<canvas" in page
    assert "view game" not in page
    assert "Main leaderboard" in page
    assert '"snake": [[4, 4]]' in page


def _results() -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            "example",
            "snake-simple-8x8",
            8.0,
            False,
            20,
            {"max_length": 64, "seed_moves": [10, 20]},
        ),
    ]


def test_game_length_text() -> None:
    result = BenchmarkResult(
        "example", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64, "seed_moves": [10, 20]}
    )
    assert _game_length_text(result) == "15.0 ± 5.0"
    missing = BenchmarkResult("example", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64})
    assert _game_length_text(missing) == "-"


def test_render_site_writes_game_pages(tmp_path: Path) -> None:
    payload = record_game(_right_model())
    index = render_site(_results(), tmp_path, {"snake-simple-8x8--example": payload})
    html = index.read_text()
    assert "view game" in html
    assert "<th>Game length</th>" in html
    assert "15.0 ± 5.0" in html
    assert "<th>Replay</th>" in html
    assert 'href="games/snake-simple-8x8--example.html"' in html
    game_page = tmp_path / "games" / "snake-simple-8x8--example.html"
    assert game_page.is_file()
    assert "Main leaderboard" in game_page.read_text()
    raw = json.loads((tmp_path / "games" / "snake-simple-8x8--example.json").read_text())
    assert raw["seed"] == REPLAY_SEED


def test_render_site_without_replays_has_no_replay_column(tmp_path: Path) -> None:
    index = render_site(_results(), tmp_path)
    html = index.read_text()
    assert "<th>Replay</th>" not in html
    assert "<th>Game length</th>" in html
    assert "view game" not in html
    assert not (tmp_path / "games").exists()


def test_render_game_pages_uses_slug_lookup(tmp_path: Path) -> None:
    payload = record_game(_right_model())
    result = BenchmarkResult("my model", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64})
    links = render_game_pages([result], {"snake-simple-8x8--my_model": payload}, tmp_path)
    assert links == {("snake-simple-8x8", "my model"): "games/snake-simple-8x8--my_model.html"}
    assert (tmp_path / "games" / "snake-simple-8x8--my_model.html").is_file()


def test_render_game_pages_rejects_slug_collisions(tmp_path: Path) -> None:
    payload = record_game(_right_model())
    results = [
        BenchmarkResult("my model", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64}),
        BenchmarkResult("my_model", "snake-simple-8x8", 8.0, False, 20, {"max_length": 64}),
    ]
    with pytest.raises(ValueError, match="collision"):
        render_game_pages(results, {"snake-simple-8x8--my_model": payload}, tmp_path)


def test_render_site_removes_stale_game_pages(tmp_path: Path) -> None:
    stale = tmp_path / "games" / "stale.html"
    stale.parent.mkdir()
    stale.write_text("stale")
    payload = record_game(_right_model())
    render_site(_results(), tmp_path, {"snake-simple-8x8--example": payload})
    assert not stale.exists()


def test_collect_reports_replay_dir_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fast_snake_cli: None
) -> None:
    import shutil

    models = tmp_path / "models"
    shutil.copytree(
        Path("models/snake-simple-8x8/example"), models / "snake-simple-8x8" / "example"
    )
    replays = tmp_path / "replays"
    replays.write_text("not a directory")
    assert (
        main(
            [
                "collect",
                "--models",
                str(models),
                "--output",
                str(tmp_path / "leaderboard.json"),
                "--replay-dir",
                str(replays),
            ]
        )
        == 2
    )
    assert "replays" in caplog.text


def test_collect_and_render_replay_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fast_snake_cli: None
) -> None:
    import shutil

    models = tmp_path / "models"
    shutil.copytree(
        Path("models/snake-simple-8x8/example"), models / "snake-simple-8x8" / "example"
    )
    output = tmp_path / "leaderboard.json"
    replays = tmp_path / "replays"
    assert (
        main(
            [
                "collect",
                "--models",
                str(models),
                "--output",
                str(output),
                "--replay-dir",
                str(replays),
            ]
        )
        == 0
    )
    assert (replays / "snake-simple-8x8--example.json").is_file()
    docs = tmp_path / "docs"
    assert main(["render", str(output), str(docs), "--replays", str(replays)]) == 0
    assert (docs / "games" / "snake-simple-8x8--example.html").is_file()
    assert "view game" in (docs / "index.html").read_text()
    capsys.readouterr()


def test_collect_records_replay_for_selected_input_indices(
    tmp_path: Path, fast_snake_cli: None
) -> None:
    directory = tmp_path / "models" / "snake-simple-8x8" / "selected"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "snake-simple-8x8",
                "name": "selected",
                "author": "tester",
                "architecture": "linear",
                "input_size": 2,
                "output_size": 4,
                "input_indices": [0, 1],
            }
        )
    )
    torch.save({"weight": torch.zeros((4, 2)), "bias": torch.zeros(4)}, directory / "model.pt")
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
    assert (replays / "snake-simple-8x8--selected.json").is_file()


def _reference_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[index] = gae
    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, returns


def test_snake_training_writes_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_nn_competition.examples import snake_submission

    monkeypatch.setattr(snake_submission, "_evaluate", lambda *args: [1])
    output = tmp_path / "submission"
    state = snake_submission.train(
        output,
        iterations=1,
        episodes_per_iteration=1,
        max_moves=8,
        update_epochs=1,
        device_name="cpu",
    )
    config = json.loads((output / "config.json").read_text())
    assert config["benchmark"] == "snake-simple-8x8"
    assert config["name"] == "submission"
    assert config["hidden_sizes"] == [64, 64]
    assert state.keys() == torch.load(output / "model.pt", weights_only=True).keys()


def test_gae_matches_literal_reference() -> None:
    import random

    from minimal_nn_competition.examples.snake_submission import _gae

    rng = random.Random(7)
    for length in range(11):
        rewards = [rng.uniform(-1.0, 1.0) for _ in range(length)]
        values = [rng.uniform(-1.0, 1.0) for _ in range(length)]
        dones = [rng.random() < 0.25 for _ in range(length)]
        expected_advantages, expected_returns = _reference_gae(rewards, values, dones, 0.99, 0.95)
        advantages, returns = _gae(rewards, values, dones, 0.99, 0.95)
        assert advantages == expected_advantages
        assert returns == expected_returns


def test_gae_keeps_strict_length_validation() -> None:
    from minimal_nn_competition.examples.snake_submission import _gae

    with pytest.raises(ValueError):
        _gae([0.5, -1.0], [0.1, 0.2, 0.3], [False, True], 0.99, 0.95)
