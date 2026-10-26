import json
import logging
import statistics
from pathlib import Path
from typing import Any

import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.benchmarks.common.snake import benchmark as snake_benchmark
from minimal_nn_competition.benchmarks.common.snake.benchmark import DEFAULT_SEEDS
from minimal_nn_competition.benchmarks.common.snake.engine import DIRECTIONS, SnakeEngine
from minimal_nn_competition.benchmarks.common.snake.layout import (
    _check_tour,
    find_closed_tour,
    obstacles_for_seed,
    validate_seed,
)
from minimal_nn_competition.benchmarks.common.snake.observation import observe
from minimal_nn_competition.benchmarks.common.snake.replay import record_game, replay_stem
from minimal_nn_competition.benchmarks.registry import create_benchmark
from minimal_nn_competition.benchmarks.snake_obstacles_8x8 import SnakeObstacles8x8Benchmark
from minimal_nn_competition.leaderboard.render import (
    _score_text,
    render_site,
    render_submission_html,
    render_submission_pages,
)
from minimal_nn_competition.model.entry import (
    Submission,
    materialize_submission,
)
from minimal_nn_competition.runner.entry import BenchmarkResult

FAST_SEEDS = (0, 1, 2)


def _write_submission(tmp_path: Path) -> Submission:
    directory = tmp_path / "example"
    directory.mkdir()
    config = {
        "benchmark": "snake-obstacles-8x8",
        "name": "example",
        "author": "tester",
        "architecture": "linear",
        "input_size": 24,
        "output_size": 4,
    }
    (directory / "config.json").write_text(json.dumps(config))
    torch.save({"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)}, directory / "model.pt")
    return Submission.from_directory(directory)


def _right_model() -> object:
    class Right:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    return Right()


def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


def _action_between(first: tuple[int, int], second: tuple[int, int]) -> int:
    return DIRECTIONS.index((second[0] - first[0], second[1] - first[1]))


def _follow_to_full(engine: SnakeEngine, tour: tuple[tuple[int, int], ...]) -> None:
    positions = {cell: index for index, cell in enumerate(tour)}
    while len(engine.snake) < engine.free_count:
        current = engine.snake[0]
        next_cell = tour[(positions[current] + 1) % len(tour)]
        _, done = engine.step(_action_between(current, next_cell))
        assert done is False


def test_obstacles_are_deterministic_and_use_two_or_four_cells() -> None:
    counts = set()
    for seed in range(10):
        blocked = obstacles_for_seed(8, seed)
        assert blocked == obstacles_for_seed(8, seed)
        assert len(blocked) in (2, 4)
        assert len(set(blocked)) == len(blocked)
        assert (4, 4) not in blocked
        assert all(0 <= x < 8 and 0 <= y < 8 for x, y in blocked)
        counts.add(len(blocked))
    assert counts == {2, 4}


def test_closed_tour_visits_every_free_cell_without_going_left() -> None:
    for seed in (*FAST_SEEDS, 4):
        blocked = obstacles_for_seed(8, seed)
        tour = find_closed_tour(8, blocked)
        assert tour is not None
        free = {(x, y) for y in range(8) for x in range(8)} - set(blocked)
        assert tour[0] == (4, 4)
        assert len(tour) == len(free) == len(set(tour))
        assert set(tour) == free
        assert all(_adjacent(first, second) for first, second in zip(tour, tour[1:], strict=False))
        assert _adjacent(tour[-1], tour[0])
        assert tour[1] != (3, 4)


def test_validate_seed_proves_the_game_is_passable() -> None:
    for seed in FAST_SEEDS:
        result = validate_seed(8, seed)
        assert result["length"] == 64 - len(result["blocked"])
        assert result["tour_length"] == result["length"]
        assert result["moves"] <= 4096


def test_food_never_spawns_on_blocked_cells() -> None:
    blocked = ((4, 0), (5, 0))
    engine = SnakeEngine(size=8, seed=3, blocked=blocked)
    for _ in range(200):
        engine._place_food()
        assert engine.food not in blocked


def test_moving_into_a_blocked_cell_loses() -> None:
    engine = SnakeEngine(size=8, seed=0, blocked=((5, 4), (0, 0)))
    engine.reset()
    assert engine.step(1) == (-1.0, True)
    assert engine.won is False


def test_full_board_waits_for_the_extra_loop_move() -> None:
    seed = 0
    blocked = obstacles_for_seed(8, seed)
    tour = find_closed_tour(8, blocked)
    assert tour is not None
    engine = SnakeEngine(size=8, seed=seed, max_moves=4096, blocked=blocked, loop_on_full=True)
    engine.reset()
    _follow_to_full(engine, tour)
    assert len(engine.snake) == engine.free_count
    assert engine.won is False
    assert engine.food is None
    positions = {cell: index for index, cell in enumerate(tour)}
    tail = tour[(positions[engine.snake[0]] + 1) % len(tour)]
    reward, done = engine.step(_action_between(engine.snake[0], tail))
    assert (reward, done) == (0.0, True)
    assert engine.won is True


def test_full_board_bad_extra_move_loses() -> None:
    seed = 0
    blocked = obstacles_for_seed(8, seed)
    tour = find_closed_tour(8, blocked)
    assert tour is not None
    engine = SnakeEngine(size=8, seed=seed, max_moves=4096, blocked=blocked, loop_on_full=True)
    engine.reset()
    _follow_to_full(engine, tour)
    current = engine.snake[0]
    positions = {cell: index for index, cell in enumerate(tour)}
    tail = tour[(positions[current] + 1) % len(tour)]
    tail_direction = (tail[0] - current[0], tail[1] - current[1])
    reverse = (-engine.direction[0], -engine.direction[1])
    action = next(
        index
        for index, direction in enumerate(DIRECTIONS)
        if direction not in (tail_direction, reverse)
    )
    reward, done = engine.step(action)
    assert (reward, done) == (-1.0, True)
    assert engine.won is False
    assert len(engine.snake) == engine.free_count


def test_observation_rays_include_blocked_cells() -> None:
    engine = SnakeEngine(size=8, seed=0, blocked=((6, 4), (0, 0)))
    engine.set_snake([(4, 4)])
    engine.food = (0, 7)
    values = observe(engine)
    assert values[16] == pytest.approx(5 / 6)

    plain = SnakeEngine(size=8, seed=0)
    plain.set_snake([(4, 4)])
    plain.food = (0, 7)
    assert observe(plain)[16] == 0.0


def test_obstacle_benchmark_is_registered() -> None:
    benchmark = create_benchmark("snake-obstacles-8x8")
    assert isinstance(benchmark, SnakeObstacles8x8Benchmark)
    assert benchmark.name == "snake-obstacles-8x8"
    assert benchmark.size == 8
    assert benchmark.loop_on_full is True
    assert benchmark.seeds == DEFAULT_SEEDS


def test_obstacle_benchmark_evaluation_with_dummy_model(tmp_path: Path) -> None:
    submission = _write_submission(tmp_path)
    score, passed, details = SnakeObstacles8x8Benchmark(seeds=FAST_SEEDS).evaluate(
        materialize_submission(submission)
    )
    assert score == statistics.fmean(details["seed_passed"])
    assert passed is False
    assert details["seed_target_lengths"] == [
        64 - len(obstacles_for_seed(8, seed)) for seed in FAST_SEEDS
    ]
    assert details["seed_blocked"] == [
        [list(cell) for cell in obstacles_for_seed(8, seed)] for seed in FAST_SEEDS
    ]
    assert details["seed_passed"] == [False, False, False]


@pytest.mark.parametrize("won", [True, False])
def test_obstacle_benchmark_requires_the_loop_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, won: bool
) -> None:
    class LoopEnvironment:
        def __init__(
            self,
            size: int,
            seed: int,
            max_moves: int,
            input_indices: object = None,
            blocked: object = (),
            loop_on_full: bool = False,
        ) -> None:
            self.size = size
            self.won = won
            self.snake = [(0, 0)] * (size * size - len(blocked))

        def reset(self) -> tuple[float, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", LoopEnvironment)
    submission = _write_submission(tmp_path)
    _, passed, details = SnakeObstacles8x8Benchmark(seeds=FAST_SEEDS).evaluate(
        materialize_submission(submission)
    )
    assert passed is won
    assert details["seed_passed"] == [won, won, won]


def test_replay_stem_prefixes_non_default_snake_benchmarks() -> None:
    assert replay_stem("snake-simple-8x8", "my model") == "snake-simple-8x8--my_model"
    assert replay_stem("snake-simple-16x16", "my model") == "snake-simple-16x16--my_model"
    assert replay_stem("snake-obstacles-8x8", "my model") == "snake-obstacles-8x8--my_model"


def test_replay_payload_and_render_include_obstacles(tmp_path: Path) -> None:
    blocked = obstacles_for_seed(8, 0)
    payload = record_game(
        _right_model(),
        seed=0,
        blocked=blocked,
        loop_on_full=True,
    )
    assert payload["blocked"] == [list(cell) for cell in blocked]
    result = BenchmarkResult(
        "example",
        "snake-obstacles-8x8",
        60.0,
        True,
        20,
        {"seed_lengths": [60], "seed_target_lengths": [60], "seed_moves": [100]},
    )
    page = render_submission_html(result, payload, 1, 1)
    assert "Obstacles 8×8" in page
    assert "Replay" in page
    assert "const BLOCKED" in page
    assert "if (food)" in page
    links = render_submission_pages([result], {"snake-obstacles-8x8--example": payload}, tmp_path)
    assert links == {
        ("snake-obstacles-8x8", "example"): "submissions/snake-obstacles-8x8--example.html"
    }
    rendered = (tmp_path / "submissions" / "snake-obstacles-8x8--example.html").read_text()
    assert "const BLOCKED" in rendered


def test_replay_render_handles_a_full_board_without_food() -> None:
    payload = {
        "seed": 0,
        "size": 8,
        "blocked": [[0, 0], [1, 0]],
        "length": 62,
        "moves": 1,
        "frames": [{"snake": [[4, 4]], "food": None}],
    }
    result = BenchmarkResult(
        "example", "snake-obstacles-8x8", 1.0, True, 20, {"seed_lengths": [60]}
    )
    page = render_submission_html(result, payload, 1, 1)
    assert '"food": null' in page
    assert "if (food)" in page


def test_score_text_renders_pass_fraction() -> None:
    result = BenchmarkResult(
        "example",
        "snake-obstacles-8x8",
        0.5,
        False,
        20,
        {"seed_lengths": [31, 30], "seed_target_lengths": [62, 60], "seed_passed": [True, False]},
    )
    assert _score_text(result) == "50.00%"


def test_render_site_shows_the_obstacle_section(tmp_path: Path) -> None:
    blocked = obstacles_for_seed(8, 0)
    payload = record_game(_right_model(), seed=0, blocked=blocked, loop_on_full=True)
    result = BenchmarkResult(
        "example",
        "snake-obstacles-8x8",
        0.5,
        False,
        20,
        {"seed_lengths": [31], "seed_target_lengths": [62], "seed_moves": [10]},
    )
    index = render_site([result], tmp_path, {"snake-obstacles-8x8--example": payload})
    html = index.read_text()
    assert "Obstacles 8×8" in html
    assert "50.00%" in html
    assert "0.00%" in html


def test_validate_cli_passes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert main(["validate-snake-obstacles-8x8", "--seeds", "3"]) == 0
    assert "seed 0 passed" in caplog.text
    assert "seed 2 passed" in caplog.text


def test_train_obstacle_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_nn_competition.examples import snake_submission

    calls = []

    def fake_train(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(snake_submission, "train", fake_train)
    assert main(["train-snake-obstacles-8x8", "--output", str(tmp_path / "output")]) == 0
    assert calls[0][1]["benchmark"] == "snake-obstacles-8x8"


def test_train_obstacle_writes_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minimal_nn_competition.examples import snake_submission

    def cheap_evaluate(policy: object, device: object, seeds: object, benchmark: Any) -> list[int]:
        return [1]

    monkeypatch.setattr(snake_submission, "_evaluate", cheap_evaluate)
    output = tmp_path / "submission"
    snake_submission.train(
        output,
        iterations=1,
        episodes_per_iteration=1,
        max_moves=8,
        update_epochs=1,
        device_name="cpu",
        benchmark="snake-obstacles-8x8",
    )
    config = json.loads((output / "config.json").read_text())
    assert config["benchmark"] == "snake-obstacles-8x8"


def test_all_default_seeds_have_verified_tours() -> None:
    for seed in range(100):
        result = validate_seed(8, seed)
        assert result["tour_length"] == 64 - len(result["blocked"])
        assert result["moves"] <= 4096


def test_observe_handles_a_full_board_without_food() -> None:
    blocked = obstacles_for_seed(8, 0)
    tour = find_closed_tour(8, blocked)
    assert tour is not None
    engine = SnakeEngine(size=8, seed=0, max_moves=4096, blocked=blocked, loop_on_full=True)
    engine.reset()
    _follow_to_full(engine, tour)
    values = observe(engine)
    assert engine.food is None
    assert len(values) == 24
    assert all(isinstance(value, float) for value in values)


def test_full_board_body_collision_loses() -> None:
    path = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (2, 1),
        (1, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (0, 3),
    ]
    engine = SnakeEngine(size=4, seed=0, max_moves=100, loop_on_full=True)
    engine.set_snake(path)
    engine.food_bit = None
    assert len(engine.snake) == engine.free_count == 16
    assert engine.step(1) == (-1.0, True)
    assert engine.won is False


def test_max_moves_cap_prevents_the_bonus_loop_win() -> None:
    seed = 0
    baseline = validate_seed(8, seed)
    blocked = obstacles_for_seed(8, seed)
    tour = find_closed_tour(8, blocked)
    assert tour is not None
    free_count = 64 - len(blocked)
    engine = SnakeEngine(
        size=8,
        seed=seed,
        max_moves=baseline["moves"] - 1,
        blocked=blocked,
        loop_on_full=True,
    )
    engine.reset()
    positions = {cell: index for index, cell in enumerate(tour)}
    done = False
    while not done:
        current = engine.snake[0]
        next_cell = tour[(positions[current] + 1) % len(tour)]
        _, done = engine.step(_action_between(current, next_cell))
    assert len(engine.snake) == free_count
    assert engine.won is False


def test_find_closed_tour_rejects_impossible_layouts_and_lists() -> None:
    assert find_closed_tour(8, [(0, 0), (0, 2)]) is None


def test_engine_rejects_out_of_bounds_or_centre_blocked_cells() -> None:
    with pytest.raises(ValueError, match="inside the board"):
        SnakeEngine(size=8, blocked=((8, 0),))
    with pytest.raises(ValueError, match="start cell"):
        SnakeEngine(size=8, blocked=((4, 4),))


def test_check_tour_rejects_an_incomplete_tour() -> None:
    blocked = obstacles_for_seed(8, 0)
    tour = find_closed_tour(8, blocked)
    assert tour is not None
    with pytest.raises(AssertionError):
        _check_tour(8, blocked, tour[:-1])


def test_replay_stems_are_distinct_for_the_same_submission_name() -> None:
    stems = {
        replay_stem(benchmark, "example")
        for benchmark in (
            "snake-simple-8x8",
            "snake-simple-16x16",
            "snake-obstacles-8x8",
        )
    }
    assert len(stems) == 3
