import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import minimal_nn_competition.__main__ as cli
from minimal_nn_competition.benchmarks.common import control
from minimal_nn_competition.benchmarks.registry import RL_BENCHMARKS, create_benchmark
from minimal_nn_competition.examples import control_submission
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission
from minimal_nn_competition.runner.entry import BenchmarkResult, evaluate_submission


def zero_submission(root, name):
    task = control.TASKS[name]
    directory = root / name / "zero"
    directory.mkdir(parents=True)
    config = dict(
        benchmark=name,
        name="zero",
        author="tester",
        architecture="linear",
        input_size=task["inputs"],
        output_size=task["outputs"],
    )
    (directory / "config.json").write_text(json.dumps(config))
    model = torch.nn.Linear(task["inputs"], task["outputs"])
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model.state_dict(), directory / "model.pt")
    return Submission.from_directory(directory)


def test_registry_includes_control_benchmarks():
    assert set(RL_BENCHMARKS) == set(control.TASKS)
    for name in ("cartpole", "pendulum", "lunarlander"):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            create_benchmark(name)


@pytest.mark.parametrize("name", RL_BENCHMARKS)
def test_reset_bounds_and_repeatable_seed_variation(name):
    task = control.TASKS[name]
    with control.environment(name) as env:
        states = [env.reset(seed=seed, options=task["reset"])[0] for seed in range(100)]
        for seed, initial in enumerate(states):
            np.testing.assert_array_equal(initial, env.reset(seed=seed, options=task["reset"])[0])
    assert len({tuple(state) for state in states}) == 100
    if name.startswith("cartpole"):
        assert np.abs(states).max() <= 0.1
        assert max(abs(state[2]) for state in states) < math.radians(12)
    if name.startswith("pendulum"):
        assert max(abs(state[2]) for state in states) <= 2
        assert max(abs(state[2]) for state in states) > 1


@pytest.mark.parametrize("name", RL_BENCHMARKS)
def test_native_transitions_rewards_and_termination(name, monkeypatch):
    task = control.TASKS[name]
    with control.environment(name) as adapted, gym.make(task["environment"]) as native:
        if name.startswith("lunarlander"):
            # Compare from identical Box2D reset states; no extra-force hook after reset.
            native.close()
            native = control.environment(name)
        try:
            np.testing.assert_array_equal(
                adapted.reset(seed=7, options=task["reset"])[0],
                native.reset(seed=7, options=task["reset"])[0],
            )
            if name.startswith("lunarlander"):
                base = native.unwrapped
                assert len(list(base.world.joints)) == 2
                state = adapted.reset(seed=7, options=task["reset"])[0]
                assert state[2] == np.float32(adapted.unwrapped.lander.linearVelocity.x * 10 / 50)
                shaping = (
                    -100 * np.linalg.norm(state[:2].astype(float))
                    - 100 * np.linalg.norm(state[2:4].astype(float))
                    - 100 * abs(float(state[4]))
                    + 10 * sum(state[6:])
                )
                assert base.prev_shaping == pytest.approx(shaping, abs=1e-4)
                monkeypatch.setattr(base, "step", control.LunarLander.step.__get__(base))
            for _index in range(task["horizon"]):
                action = control.action(name, np.arange(task["outputs"]) + 0.5)
                a, b = adapted.step(action), native.step(action)
                np.testing.assert_array_equal(a[0], b[0])
                assert a[1:] == b[1:]
                if a[2] or a[3]:
                    break
            if name.startswith("pendulum"):
                assert _index + 1 == 200 and a[3] and not a[2]
        finally:
            native.close()


def test_continuous_action_validation(tmp_path):
    name = "pendulum-gym-v1"
    for value, expected in [(-3, -2), (0.25, 0.25), (3, 2)]:
        action = control.action(name, [value])
        assert action.shape == (1,) and action.dtype == np.float32
        assert action[0] == expected
    for bad in ([float("nan")], [float("inf")], [], [1, 2], [[1]]):
        with pytest.raises(ValueError):
            control.action(name, bad)
    assert control.action("lunarlander-gym-v1", [0, 0, 1, 0]) == 2
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.validation import validate_config

    submission = zero_submission(tmp_path, name)
    config_path = submission.checkpoint.with_name("config.json")
    config = json.loads(config_path.read_text())
    config["output_size"] = 5
    config_path.write_text(json.dumps(config))
    with pytest.raises(SubmissionError, match="output_size 1"):
        Submission.from_directory(config_path.parent)
    assert validate_config({**config, "benchmark": []}, Path("models") / name / "zero")
    assert any(
        "output_size 1" in error
        for error in validate_config(config, Path("models") / name / "zero")
    )


@pytest.mark.parametrize("name", RL_BENCHMARKS)
def test_evaluation_is_deterministic_and_fingerprints_exact_contract(tmp_path, name):
    benchmark = create_benchmark(name, seeds=(0, 1, 2))
    submission = zero_submission(tmp_path, name)
    first = evaluate_submission(submission, benchmark)
    second = evaluate_submission(submission, benchmark)
    assert first == second
    details = first.details
    manifest = details["evaluation_spec"]
    assert (
        details["evaluation_sha256"]
        == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    )
    assert details["seed_returns"] and len(details["trajectory_sha256"]) == 3
    if name.startswith("pendulum"):
        assert first.parameter_count == 4
        assert details["seed_steps"] == [200] * 3
        assert details["seed_truncated"] == [True] * 3
    assert first.score == sum(details["seed_success"]) / len(details["seed_success"])
    assert first.passed is all(details["seed_success"])
    changed = evaluate_submission(submission, create_benchmark(name, seeds=(0,)))
    assert changed.details["evaluation_sha256"] != details["evaluation_sha256"]
    wrong = Submission(
        config=submission.config.model_copy(update={"output_size": 5}),
        checkpoint=submission.checkpoint,
    )
    with pytest.raises(ValueError, match="output_size"):
        benchmark.load_submission(wrong)


def test_cartpole_last_step_failure_and_lander_landing_fraction(tmp_path, monkeypatch):
    episode = dict(
        reward=500,
        steps=500,
        terminated=True,
        truncated=True,
        landed=False,
        trajectory_sha256="test",
    )
    monkeypatch.setattr(control, "rollout", lambda *a: episode)
    result = evaluate_submission(
        zero_submission(tmp_path, "cartpole-gym-v1"),
        create_benchmark("cartpole-gym-v1", seeds=(0,)),
    )
    assert result.score == 0 and not result.passed
    successes = iter((False, True))
    monkeypatch.setattr(
        control, "rollout", lambda *a: {**episode, "reward": 300, "landed": next(successes)}
    )
    result = evaluate_submission(
        zero_submission(tmp_path, "lunarlander-gym-v1"),
        create_benchmark("lunarlander-gym-v1", seeds=(0, 1)),
    )
    assert result.score == 0.5 and not result.passed
    assert result.details["mean_return"] == 300


@pytest.fixture
def fast_control_cli(monkeypatch):
    monkeypatch.setattr(
        cli, "create_benchmark", lambda name, **kw: create_benchmark(name, seeds=(0, 1, 2), **kw)
    )


def test_control_evaluate_cli(capsys, fast_control_cli):
    for name in RL_BENCHMARKS:
        assert cli.main(["evaluate", name, f"models/{name}/example"]) == 0
        assert f'"benchmark": "{name}"' in capsys.readouterr().out


def test_control_collect_cli(tmp_path, fast_control_cli, commit_submission_date):
    models = tmp_path / "models"
    for name in RL_BENCHMARKS:
        zero_submission(models, name)
    commit_submission_date(tmp_path)
    output = tmp_path / "leaderboard.json"
    assert cli.main(["collect", "--models", str(models), "--output", str(output)]) == 0
    assert {r["benchmark"] for r in json.loads(output.read_text())} == set(RL_BENCHMARKS)


def test_control_train_commands_wire_examples(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        control_submission, "train", lambda benchmark, output, **kw: calls.append((benchmark, kw))
    )
    for name in RL_BENCHMARKS:
        assert (
            cli.main(
                [
                    f"train-{name.removesuffix('-gym-v1')}",
                    "--output",
                    str(tmp_path),
                    "--iterations",
                    "1",
                ]
            )
            == 0
        )
    assert [name for name, _ in calls] == list(RL_BENCHMARKS)
    assert all(kw["iterations"] == 1 for _, kw in calls)


def test_render_control_group(tmp_path):
    result = BenchmarkResult(
        "zero",
        "lunarlander-gym-v1",
        0.95,
        False,
        36,
        {"mean_return": 279.95, "seed_success": [True] * 95 + [False] * 5},
        name="zero",
        author="tester",
        submitted_at="2026-10-30",
    )
    page = render_site([result], tmp_path).read_text()
    assert 'id="control"' in page and "Lunar Lander" in page
    assert "95.00%" in page
    assert "Reward" not in page
    assert "279.95" not in page
    detail = (tmp_path / "submissions" / "lunarlander-gym-v1--zero.html").read_text()
    assert "Reward" not in detail
    assert "279.95" not in detail


@pytest.mark.parametrize("name", ("pendulum-gym-v1", "lunarlander-gym-v1"))
def test_native_ppo_training_writes_loadable_policy(tmp_path, name):
    control_submission.train(
        name, tmp_path / name, iterations=1, episodes_per_iteration=1, device_name="cpu"
    )
    submission = Submission.from_directory(tmp_path / name)
    result = evaluate_submission(submission, create_benchmark(name, seeds=(10000,)))
    assert math.isfinite(result.score)


def test_lander_landing_requires_safe_settled_position():
    from types import SimpleNamespace as Object

    base = Object(
        game_over=False,
        lander=Object(awake=False),
        helipad_x1=0,
        helipad_x2=2,
        legs=[Object(position=Object(x=0)), Object(position=Object(x=2))],
    )
    env = Object(unwrapped=base)
    assert control.landed(env, True)
    assert not control.landed(env, False)  # Timeout/hover is not a landing.
    base.game_over = True
    assert not control.landed(env, True)  # A crash cannot pass even inside the posts.
    base.game_over = False
    base.lander.awake = True
    assert not control.landed(env, True)
    base.lander.awake = False
    for leg, x in ((0, -0.001), (1, 2.001)):
        previous = base.legs[leg].position.x
        base.legs[leg].position.x = x
        assert not control.landed(env, True)
        base.legs[leg].position.x = previous
