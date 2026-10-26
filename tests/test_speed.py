import json
from pathlib import Path

import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.common.snake.engine import SnakeEngine
from minimal_nn_competition.speed.agent import (
    BOARD_SIZE,
    HAMILTONIAN_CYCLE,
    SIZE,
    action_for,
)
from minimal_nn_competition.speed.benchmark import SpeedResult, measure_snake_speed
from minimal_nn_competition.speed.leaderboard import (
    load_speed_results,
    save_speed_results,
    upsert_speed_result,
)
from minimal_nn_competition.speed.render import render_speed_site


def _result(commit: str, ticks: float, std: float = 0.0) -> SpeedResult:
    return SpeedResult(
        commit=commit,
        ticks_per_second=ticks,
        ticks_std=std,
        run_ticks=[int(ticks)],
        measured_at="2026-10-22T00:00:00+00:00",
        duration_seconds=1.0,
        runs=1,
        warmup_seconds=0.0,
        agent="test-agent",
        harness="test-harness",
        python_version="3.12.14",
        platform="test-platform",
        machine="arm64",
        cpu="Apple M5 Pro",
        validation_seeds=100,
    )


def test_hamiltonian_cycle_is_valid() -> None:
    assert len(HAMILTONIAN_CYCLE) == BOARD_SIZE
    assert len(set(HAMILTONIAN_CYCLE)) == BOARD_SIZE
    assert set(HAMILTONIAN_CYCLE) == {(x, y) for y in range(SIZE) for x in range(SIZE)}
    for index, (x, y) in enumerate(HAMILTONIAN_CYCLE):
        next_x, next_y = HAMILTONIAN_CYCLE[(index + 1) % BOARD_SIZE]
        assert abs(next_x - x) + abs(next_y - y) == 1


def test_hamiltonian_cycle_order_is_pinned() -> None:
    assert "".join(f"{x}{y}" for x, y in HAMILTONIAN_CYCLE) == (
        "4443424140506070717273747576776766656463626151525354555657474645"
        "3536372726252423222111121314151617070605040302010010203031323334"
    )


def test_agent_fills_board_on_every_seed() -> None:
    for seed in range(100):
        engine = SnakeEngine(size=SIZE, seed=seed, max_moves=4096)
        done = False
        while not done:
            _, done = engine.step(action_for(engine))
        assert len(engine.snake) == BOARD_SIZE


def test_measure_speed_short_run() -> None:
    result = measure_snake_speed(duration=0.01, runs=2, warmup=0.0, validate=False)
    assert result.runs == 2
    assert len(result.run_ticks) == 2
    assert all(ticks > 0 for ticks in result.run_ticks)
    assert result.ticks_per_second > 0
    assert result.ticks_std >= 0
    assert result.agent == "hamiltonian-cycle-v1"
    assert result.validation_seeds == 0


def test_measure_speed_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="duration"):
        measure_snake_speed(duration=0, validate=False)
    with pytest.raises(ValueError, match="runs"):
        measure_snake_speed(runs=0, validate=False)
    with pytest.raises(ValueError, match="warmup"):
        measure_snake_speed(warmup=-1, validate=False)


def test_speed_leaderboard_round_trip(tmp_path: Path) -> None:
    result = _result("abc1234", 1_500_000.0, 12_000.0)
    path = tmp_path / "speed-leaderboard.json"
    save_speed_results(path, [result])
    assert load_speed_results(path) == [result]


def test_speed_leaderboard_upsert_and_ranking(tmp_path: Path) -> None:
    old = _result("abc1234", 1_000_000.0)
    new = _result("abc1234", 2_000_000.0)
    faster = _result("def5678", 3_000_000.0)
    results = upsert_speed_result([old, faster], new)
    assert [result.commit for result in results] == ["def5678", "abc1234"]
    assert [result.ticks_per_second for result in results] == [3_000_000.0, 2_000_000.0]
    path = tmp_path / "speed-leaderboard.json"
    save_speed_results(path, results)
    assert [result.commit for result in load_speed_results(path)] == ["def5678", "abc1234"]


def test_render_speed_site(tmp_path: Path) -> None:
    result = _result("abc1234567890", 1_234_567.0, 12_345.0)
    index = render_speed_site([result], tmp_path)
    page = index.read_text()
    assert "Snake Simple 8x8 Environment Speed" in page
    assert "noindex,nofollow" in page
    assert "abc1234" in page
    assert "2026-10-22 00:00:00 UTC" in page
    assert "1,234,567 ± 12,345" in page
    assert "test-harness" in page
    assert "Apple M5 Pro" in page
    assert "Main leaderboard" in page
    raw = json.loads((tmp_path / "speed" / "leaderboard.json").read_text())
    assert raw[0]["commit"] == "abc1234567890"


def test_speed_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "speed-leaderboard.json"
    assert (
        main(
            [
                "speed",
                "--duration",
                "0.01",
                "--runs",
                "2",
                "--warmup",
                "0",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    assert '"ticks_per_second"' in capsys.readouterr().out
    results = load_speed_results(path)
    assert len(results) == 1
    assert results[0].commit != "unknown"
    assert results[0].measured_at
    assert results[0].validation_seeds == 100


def test_render_speed_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "speed-leaderboard.json"
    save_speed_results(path, [_result("abc1234", 1_500_000.0)])
    output = tmp_path / "docs"
    assert main(["render-speed", str(path), str(output)]) == 0
    assert (output / "speed" / "index.html").is_file()
    assert capsys.readouterr().out.strip().endswith("index.html")
