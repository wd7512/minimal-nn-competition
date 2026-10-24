from pathlib import Path

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.control.replay import (
    REPLAY_SEED,
    load_replay,
    mean_steps,
    record_game,
    replay_stem,
    save_replay,
)
from minimal_nn_competition.leaderboard.render import (
    render_site,
    render_submission_html,
    render_submission_pages,
)
from minimal_nn_competition.runner.entry import BenchmarkResult


class _FixedAction:
    def __init__(self, outputs, action):
        self.outputs = outputs
        self.action = action

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, self.outputs), -1.0)
        logits[0, self.action] = 1.0
        return logits


def _result(benchmark, details):
    return BenchmarkResult("example", benchmark, 0.0, True, 10, details)


def test_record_game_is_deterministic() -> None:
    for benchmark, outputs, action in (
        ("cartpole", 2, 0),
        ("pendulum", 5, 2),
        ("lunarlander", 4, 1),
    ):
        first = record_game(benchmark, _FixedAction(outputs, action), seed=REPLAY_SEED)
        second = record_game(benchmark, _FixedAction(outputs, action), seed=REPLAY_SEED)
        assert first == second
        assert first["kind"] == benchmark
        assert first["seed"] == REPLAY_SEED
        assert first["steps"] == len(first["frames"]) - 1


def test_record_game_rejects_unknown_benchmark() -> None:
    with pytest.raises(ValueError, match="Unknown control benchmark"):
        record_game("chess-eval", _FixedAction(2, 0))


@pytest.mark.parametrize(
    "benchmark,outputs", [("cartpole", 2), ("pendulum", 5), ("lunarlander", 4)]
)
def test_record_game_requires_torch(
    monkeypatch: pytest.MonkeyPatch, benchmark: str, outputs: int
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        record_game(benchmark, _FixedAction(outputs, 0))


def test_record_game_terminal_flags() -> None:
    cartpole = record_game("cartpole", _FixedAction(2, 0), seed=REPLAY_SEED, steps=500)
    assert cartpole["terminated"] is True
    assert cartpole["landed"] is None
    assert cartpole["steps"] < 500
    assert cartpole["steps"] == len(cartpole["frames"]) - 1
    pendulum = record_game("pendulum", _FixedAction(5, 2), seed=REPLAY_SEED, steps=5)
    assert pendulum["terminated"] is False
    assert pendulum["landed"] is None
    assert pendulum["steps"] == 5
    lander = record_game("lunarlander", _FixedAction(4, 0), seed=REPLAY_SEED, steps=300)
    assert isinstance(lander["landed"], bool)
    assert lander["terminated"] is True
    frame = lander["frames"][0]
    assert frame["action"] is None
    assert set(frame) == {"state", "action", "reward"}
    assert len(lander["frames"][1]["state"]) == 6


def test_save_and_load_replay_round_trip(tmp_path: Path) -> None:
    payload = record_game("pendulum", _FixedAction(5, 2), seed=REPLAY_SEED, steps=5)
    path = save_replay(tmp_path / "replays" / "pendulum--example.json", payload)
    assert path.is_file()
    assert load_replay(path) == payload


def test_replay_stem_matches_snake_format() -> None:
    assert replay_stem("cartpole", "my model/v2") == "cartpole--my_model_v2"


def test_mean_steps() -> None:
    cartpole = _result("cartpole", {"seed_steps": [500] * 100, "max_steps": 500, "seeds": 100})
    assert mean_steps(cartpole) == "500.0 ± 0.0"
    pendulum = _result("pendulum", {"seed_returns": [-140.0], "horizon": 200})
    assert mean_steps(pendulum) == "200 ± 0.0"
    assert mean_steps(_result("cartpole", {})) == "-"


def test_render_submission_html_embeds_control_replay() -> None:
    details = {
        "seed_steps": [500] * 100,
        "max_steps": 500,
        "seeds": 100,
    }
    payload = record_game("cartpole", _FixedAction(2, 0), seed=REPLAY_SEED, steps=8)
    page = render_submission_html(_result("cartpole", details), payload, 1, 1)
    assert "<canvas" in page
    assert 'width="480" height="480"' in page
    assert "CartPole replay" in page
    assert '"kind"' not in page
    assert "__KIND__" not in page
    assert "const KIND = " in page
    assert "Game length" in page
    assert "500.0 ± 0.0" in page
    assert "Seed games" in page
    assert ">100<" in page
    assert "Available below" in page
    assert "Back to leaderboard" in page
    assert '<option value="4" selected>4x</option>' in page
    assert "Math.max(1, Math.floor(240 / Number(speed.value)))" in page


def test_render_submission_html_lunarlander_rows() -> None:
    details = {
        "seed_steps": [40, 60],
        "seed_landed": [True, False],
        "max_steps": 300,
        "seeds": 2,
    }
    payload = record_game("lunarlander", _FixedAction(4, 0), seed=REPLAY_SEED, steps=10)
    page = render_submission_html(_result("lunarlander", details), payload, 1, 1)
    assert "<canvas" in page
    assert 'width="480" height="480"' in page
    assert "LunarLander replay" in page
    assert "Landings" in page
    assert "1/2" in page
    assert "Game length" in page


def test_render_submission_html_without_replay() -> None:
    page = render_submission_html(_result("cartpole", {"seed_steps": [8], "seeds": 1}), None, 1, 1)
    assert "<canvas" not in page
    assert "Not recorded" in page
    assert "Game length" in page


def test_render_site_writes_control_games(tmp_path: Path) -> None:
    details = {
        "seed_steps": [8],
        "max_steps": 500,
        "seeds": 1,
    }
    payload = record_game("cartpole", _FixedAction(2, 0), seed=REPLAY_SEED, steps=8)
    index = render_site([_result("cartpole", details)], tmp_path, {"cartpole--example": payload})
    assert 'href="submissions/cartpole--example.html"' in index.read_text()
    page = tmp_path / "submissions" / "cartpole--example.html"
    assert page.is_file()
    assert "CartPole replay" in page.read_text()
    raw = load_replay(tmp_path / "games" / "cartpole--example.json")
    assert raw["seed"] == REPLAY_SEED
    assert raw["kind"] == "cartpole"


def test_render_submission_pages_keeps_snake_behaviour(tmp_path: Path) -> None:
    payload = record_game("cartpole", _FixedAction(2, 0), seed=REPLAY_SEED, steps=8)
    result = _result("cartpole", {"seed_steps": [8], "seeds": 1})
    links = render_submission_pages([result], {"cartpole--example": payload}, tmp_path)
    assert links == {("cartpole", "example"): "submissions/cartpole--example.html"}
    assert (tmp_path / "submissions" / "cartpole--example.html").is_file()


def test_collect_and_render_control_replay_cli(tmp_path: Path) -> None:
    import shutil

    models = tmp_path / "models"
    for benchmark in ("cartpole", "pendulum", "lunarlander"):
        shutil.copytree(Path(f"models/{benchmark}/example"), models / benchmark / "example")
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
    for benchmark in ("cartpole", "pendulum", "lunarlander"):
        assert (replays / f"{benchmark}--example.json").is_file()
    docs = tmp_path / "docs"
    assert main(["render", str(output), str(docs), "--replays", str(replays)]) == 0
    for benchmark, label in (
        ("cartpole", "CartPole replay"),
        ("pendulum", "Pendulum replay"),
        ("lunarlander", "LunarLander replay"),
    ):
        page = docs / "submissions" / f"{benchmark}--example.html"
        assert page.is_file()
        assert label in page.read_text()
