import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

import minimal_nn_competition.__main__ as cli
from minimal_nn_competition.benchmarks import cartpole as cartpole_engine
from minimal_nn_competition.benchmarks import lunarlander as lunarlander_engine
from minimal_nn_competition.benchmarks import pendulum as pendulum_engine
from minimal_nn_competition.benchmarks.common.names import RL_BENCHMARKS
from minimal_nn_competition.benchmarks.registry import create_benchmark
from minimal_nn_competition.examples import (
    cartpole_submission,
    lunarlander_submission,
    pendulum_submission,
)
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission, materialize_submission
from minimal_nn_competition.runner.entry import BenchmarkResult, evaluate_submission

FAST_SEEDS = (0, 1, 2)


def _write_submission(
    tmp_path: Path,
    name: str,
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> Submission:
    directory = tmp_path / name
    directory.mkdir()
    payload = {"name": name, "author": "tester", **config}
    (directory / "config.json").write_text(json.dumps(payload))
    torch.save(state, directory / "model.pt")
    return Submission.from_directory(directory)


def _zero_submission(tmp_path: Path, benchmark: str, input_size: int, output_size: int):
    return _write_submission(
        tmp_path,
        f"{benchmark}-zero",
        {
            "benchmark": benchmark,
            "architecture": "linear",
            "input_size": input_size,
            "output_size": output_size,
        },
        {
            "weight": torch.zeros((output_size, input_size)),
            "bias": torch.zeros(output_size),
        },
    )


def test_cartpole_engine_is_deterministic() -> None:
    first = cartpole_engine.reset(random.Random(7))
    second = cartpole_engine.reset(random.Random(7))
    assert first == second
    first_rollout = [cartpole_engine.step(first, 1)]
    second_rollout = [cartpole_engine.step(second, 1)]
    assert first_rollout == second_rollout


def test_cartpole_reset_bounds() -> None:
    for seed in range(20):
        x, x_dot, theta, theta_dot = cartpole_engine.reset(random.Random(seed))
        assert -cartpole_engine.INITIAL_X_LIMIT <= x <= cartpole_engine.INITIAL_X_LIMIT
        assert (
            -cartpole_engine.INITIAL_VELOCITY_LIMIT
            <= x_dot
            <= cartpole_engine.INITIAL_VELOCITY_LIMIT
        )
        assert -cartpole_engine.INITIAL_THETA_LIMIT <= theta <= cartpole_engine.INITIAL_THETA_LIMIT
        assert (
            -cartpole_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
            <= theta_dot
            <= cartpole_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
        )


def test_cartpole_seeds_vary_position_and_angle() -> None:
    states = [cartpole_engine.reset(random.Random(seed)) for seed in range(100)]
    assert len({state[0] for state in states}) == 100
    assert len({state[2] for state in states}) == 100
    assert pytest.approx(math.radians(40.0)) == cartpole_engine.THETA_LIMIT


def test_pendulum_engine_is_deterministic() -> None:
    first = pendulum_engine.reset(random.Random(7))
    second = pendulum_engine.reset(random.Random(7))
    assert first == second
    assert pendulum_engine.step(first, 1) == pendulum_engine.step(second, 1)
    assert pendulum_engine.observation(first) == pendulum_engine.observation(second)


def test_pendulum_reset_bounds() -> None:
    for seed in range(20):
        theta, theta_dot = pendulum_engine.reset(random.Random(seed))
        assert -pendulum_engine.INITIAL_ANGLE_LIMIT <= theta <= pendulum_engine.INITIAL_ANGLE_LIMIT
        assert (
            -pendulum_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
            <= theta_dot
            <= pendulum_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
        )
        cos_theta, sin_theta, velocity = pendulum_engine.observation((theta, theta_dot))
        assert cos_theta == pytest.approx(math.cos(theta))
        assert sin_theta == pytest.approx(math.sin(theta))
        assert velocity == theta_dot


def test_pendulum_seeds_vary_initial_state() -> None:
    states = [pendulum_engine.reset(random.Random(seed)) for seed in range(100)]
    assert len(set(states)) == 100


def test_lunarlander_engine_is_deterministic() -> None:
    first = lunarlander_engine.reset(random.Random(7))
    second = lunarlander_engine.reset(random.Random(7))
    assert first == second
    assert lunarlander_engine.step(first, 1) == lunarlander_engine.step(second, 1)
    assert lunarlander_engine.observation(first) == lunarlander_engine.observation(second)


def test_lunarlander_reset_bounds() -> None:
    for seed in range(20):
        x, y, vx, vy, angle, angular_velocity = lunarlander_engine.reset(random.Random(seed))
        assert -lunarlander_engine.INITIAL_X_LIMIT <= x <= lunarlander_engine.INITIAL_X_LIMIT
        assert lunarlander_engine.INITIAL_Y_MIN <= y <= lunarlander_engine.INITIAL_Y_MAX
        assert (
            -lunarlander_engine.INITIAL_VELOCITY_LIMIT
            <= vx
            <= lunarlander_engine.INITIAL_VELOCITY_LIMIT
        )
        assert (
            -lunarlander_engine.INITIAL_VELOCITY_LIMIT
            <= vy
            <= lunarlander_engine.INITIAL_VELOCITY_LIMIT
        )
        assert (
            -lunarlander_engine.INITIAL_ANGLE_LIMIT
            <= angle
            <= lunarlander_engine.INITIAL_ANGLE_LIMIT
        )
        assert (
            -lunarlander_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
            <= angular_velocity
            <= lunarlander_engine.INITIAL_ANGULAR_VELOCITY_LIMIT
        )
        assert len(lunarlander_engine.observation((x, y, vx, vy, angle, angular_velocity))) == 8


def test_lunarlander_seeds_vary_initial_state() -> None:
    states = [lunarlander_engine.reset(random.Random(seed)) for seed in range(100)]
    assert len(set(states)) == 100


def test_registry_includes_control_benchmarks() -> None:
    for benchmark in RL_BENCHMARKS:
        assert create_benchmark(benchmark).name == benchmark


def test_cartpole_rejects_wrong_config(tmp_path: Path) -> None:
    wrong_input = _write_submission(
        tmp_path,
        "cartpole-input",
        {
            "benchmark": "cartpole",
            "architecture": "linear",
            "input_size": 3,
            "output_size": 2,
        },
        {"weight": torch.zeros((2, 3)), "bias": torch.zeros(2)},
    )
    with pytest.raises(ValueError, match="input_size 4"):
        create_benchmark("cartpole").load_submission(wrong_input)
    wrong_output = _write_submission(
        tmp_path,
        "cartpole-output",
        {
            "benchmark": "cartpole",
            "architecture": "linear",
            "input_size": 4,
            "output_size": 3,
        },
        {"weight": torch.zeros((3, 4)), "bias": torch.zeros(3)},
    )
    with pytest.raises(ValueError, match="output_size 2"):
        create_benchmark("cartpole").load_submission(wrong_output)


def test_pendulum_rejects_wrong_config(tmp_path: Path) -> None:
    wrong_input = _write_submission(
        tmp_path,
        "pendulum-input",
        {
            "benchmark": "pendulum",
            "architecture": "linear",
            "input_size": 4,
            "output_size": 5,
        },
        {"weight": torch.zeros((5, 4)), "bias": torch.zeros(5)},
    )
    with pytest.raises(ValueError, match="input_size 3"):
        create_benchmark("pendulum").load_submission(wrong_input)
    wrong_output = _write_submission(
        tmp_path,
        "pendulum-output",
        {
            "benchmark": "pendulum",
            "architecture": "linear",
            "input_size": 3,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 3)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="output_size 5"):
        create_benchmark("pendulum").load_submission(wrong_output)


def test_lunarlander_rejects_wrong_config(tmp_path: Path) -> None:
    wrong_input = _write_submission(
        tmp_path,
        "lunarlander-input",
        {
            "benchmark": "lunarlander",
            "architecture": "linear",
            "input_size": 7,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 7)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_size 8"):
        create_benchmark("lunarlander").load_submission(wrong_input)
    wrong_output = _write_submission(
        tmp_path,
        "lunarlander-output",
        {
            "benchmark": "lunarlander",
            "architecture": "linear",
            "input_size": 8,
            "output_size": 5,
        },
        {"weight": torch.zeros((5, 8)), "bias": torch.zeros(5)},
    )
    with pytest.raises(ValueError, match="output_size 4"):
        create_benchmark("lunarlander").load_submission(wrong_output)


def test_cartpole_calibration(tmp_path: Path) -> None:
    benchmark = create_benchmark("cartpole", seeds=FAST_SEEDS)
    random_steps = []
    for seed in FAST_SEEDS:
        state = cartpole_engine.reset(random.Random(seed))
        actions = random.Random(seed + 100)
        steps = 0
        for _ in range(500):
            state, _, done = cartpole_engine.step(state, actions.randrange(2))
            steps += 1
            if done:
                break
        random_steps.append(steps)
    assert max(random_steps) < benchmark.max_steps

    zero = materialize_submission(_zero_submission(tmp_path, "cartpole", 4, 2))
    score, passed, _ = benchmark.evaluate(zero)
    assert 0.0 <= score < 1.0
    assert passed is False

    example = Submission.from_directory(Path("models/cartpole/example"))
    result = evaluate_submission(example, benchmark)
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["seed_steps"] == [benchmark.max_steps] * len(benchmark.seeds)


def test_pendulum_calibration(tmp_path: Path) -> None:
    benchmark = create_benchmark("pendulum", seeds=FAST_SEEDS)
    returns = []
    for seed in FAST_SEEDS:
        rng = random.Random(seed)
        state = pendulum_engine.reset(rng)
        total = 0.0
        for _ in range(benchmark.max_steps):
            state, reward, _ = pendulum_engine.step(state, rng.randrange(5))
            total += reward
        returns.append(total)
    assert sum(returns) / len(returns) < benchmark.threshold

    zero = materialize_submission(_zero_submission(tmp_path, "pendulum", 3, 5))
    score, passed, _ = benchmark.evaluate(zero)
    assert 0.0 <= score < 1.0
    assert passed is False

    example = Submission.from_directory(Path("models/pendulum/example"))
    result = evaluate_submission(example, benchmark)
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["mean_return"] >= benchmark.threshold


def test_lunarlander_calibration(tmp_path: Path) -> None:
    benchmark = create_benchmark("lunarlander", seeds=FAST_SEEDS)
    returns = []
    for seed in FAST_SEEDS:
        rng = random.Random(seed)
        state = lunarlander_engine.reset(rng)
        total = 0.0
        for _ in range(benchmark.max_steps):
            state, reward, done = lunarlander_engine.step(state, rng.randrange(4))
            total += reward
            if done:
                break
        returns.append(total)
    assert sum(returns) / len(returns) < 0.0

    zero = materialize_submission(_zero_submission(tmp_path, "lunarlander", 8, 4))
    score, passed, _ = benchmark.evaluate(zero)
    assert 0.0 <= score < 1.0
    assert passed is False

    example = Submission.from_directory(Path("models/lunarlander/example"))
    result = evaluate_submission(example, benchmark)
    assert result.score == 1.0
    assert result.passed is True
    assert isinstance(result.details["mean_return"], float)


@pytest.mark.parametrize(
    ("name", "input_size", "output_size"),
    [("cartpole", 4, 2), ("pendulum", 3, 5), ("lunarlander", 8, 4)],
)
def test_control_score_is_pass_fraction(
    tmp_path: Path, name: str, input_size: int, output_size: int
) -> None:
    benchmark = create_benchmark(name, seeds=FAST_SEEDS)
    zero = materialize_submission(_zero_submission(tmp_path, name, input_size, output_size))
    score, passed, _ = benchmark.evaluate(zero)
    assert 0.0 <= score <= 1.0
    assert passed is (score == 1.0)


@pytest.fixture
def fast_control_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fast_create(name: str, **kwargs: Any) -> Any:
        if name in RL_BENCHMARKS:
            return create_benchmark(name, seeds=FAST_SEEDS, **kwargs)
        return create_benchmark(name, **kwargs)

    monkeypatch.setattr(cli, "create_benchmark", fast_create)


def test_control_evaluate_cli(capsys: pytest.CaptureFixture[str], fast_control_cli: None) -> None:
    for benchmark in RL_BENCHMARKS:
        assert main_evaluate(benchmark) == 0
        assert f'"benchmark": "{benchmark}"' in capsys.readouterr().out


def main_evaluate(benchmark: str) -> int:
    from minimal_nn_competition.__main__ import main

    return main(["evaluate", benchmark, f"models/{benchmark}/example"])


def test_control_collect_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fast_control_cli: None,
) -> None:
    import shutil

    from minimal_nn_competition.__main__ import main

    models = tmp_path / "models"
    for benchmark in RL_BENCHMARKS:
        shutil.copytree(
            Path(f"models/{benchmark}/example"),
            models / benchmark / "example",
        )
    output = tmp_path / "leaderboard.json"
    assert main(["collect", "--models", str(models), "--output", str(output)]) == 0
    results = json.loads(output.read_text())
    assert {result["benchmark"] for result in results} == set(RL_BENCHMARKS)
    assert all(result["passed"] for result in results)
    assert capsys.readouterr().out.strip() == str(output)


def test_control_train_commands_wire_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def cartpole_train(output_dir, **kwargs):
        calls.append(("cartpole", {"output_dir": output_dir, **kwargs}))
        output_dir.mkdir(parents=True)

    def pendulum_train(output_dir, **kwargs):
        calls.append(("pendulum", {"output_dir": output_dir, **kwargs}))
        output_dir.mkdir(parents=True)

    def lunarlander_train(output_dir, **kwargs):
        calls.append(("lunarlander", {"output_dir": output_dir, **kwargs}))
        output_dir.mkdir(parents=True)

    monkeypatch.setattr(cartpole_submission, "train", cartpole_train)
    monkeypatch.setattr(pendulum_submission, "train", pendulum_train)
    monkeypatch.setattr(lunarlander_submission, "train", lunarlander_train)

    from minimal_nn_competition.__main__ import main

    assert main(["train-cartpole", "--output", str(tmp_path / "cartpole")]) == 0
    assert (
        main(
            [
                "train-pendulum",
                "--output",
                str(tmp_path / "pendulum"),
                "--iterations",
                "3",
                "--episodes-per-iteration",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train-lunarlander",
                "--output",
                str(tmp_path / "lunarlander"),
                "--iterations",
                "4",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    assert [name for name, _ in calls] == ["cartpole", "pendulum", "lunarlander"]
    assert calls[0][1]["seed"] == 0
    assert calls[1][1]["iterations"] == 3
    assert calls[1][1]["episodes_per_iteration"] == 2
    assert calls[2][1]["iterations"] == 4


def test_render_control_group(tmp_path: Path) -> None:
    results = [
        BenchmarkResult("cart", "cartpole", 1.0, True, 10, {"seed_steps": [500]}),
        BenchmarkResult("pend", "pendulum", 1.0, True, 4741, {"mean_return": -140.0}),
        BenchmarkResult("land", "lunarlander", 1.0, True, 4996, {"mean_return": 8.0}),
    ]
    html = render_site(results, tmp_path).read_text()
    assert 'id="control"' in html
    assert "/ Control" in html
    assert ">CartPole<" in html
    assert ">Discrete torque<" in html
    assert ">Lunar touchdown<" in html
    assert "--accent-control" in html
    assert "prefers-color-scheme: dark" in html
    assert "100.00%" in html
    assert "Reward</th>" in html
    assert "500.0" in html
    assert "-140.00" in html
    assert "8.00" in html
