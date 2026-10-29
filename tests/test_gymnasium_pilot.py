"""Opt-in spike tests: run with the pilot's pinned dependencies (see report #229)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium", minversion="1.2.3")
pytest.importorskip("Box2D")
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pilot_gymnasium.py"
_SPEC = importlib.util.spec_from_file_location("pilot_gymnasium", _SCRIPT)
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def test_continuous_torque_shape_finiteness_and_clipping():
    for value, expected in [(-3, -2), (0.25, 0.25), (3, 2)]:
        control = pilot.action("pendulum", [value])
        assert control.shape == (1,)
        assert control.dtype == np.float32
        assert control[0] == expected
    for bad in ([float("nan")], [float("inf")], [], [1, 2], [[1]]):
        with pytest.raises(ValueError):
            pilot.action("pendulum", bad)
    assert pilot.action("lunarlander", [0, 0, 1, 0]) == 2  # Native main engine.


@pytest.mark.parametrize("task", pilot.TASKS)
def test_native_transitions_and_time_limits(task):
    # Independent native environment, same reset distribution and nonzero actions.
    with pilot.environment(task, False) as adapted, pilot.gym.make(pilot.TASKS[task][0]) as native:
        options = pilot.TASKS[task][1]
        np.testing.assert_array_equal(
            adapted.reset(seed=7, options=options)[0], native.reset(seed=7, options=options)[0]
        )
        for _ in range(1000):
            control = pilot.action(task, np.arange(pilot.TASKS[task][3]) + 0.5)
            a, b = adapted.step(control), native.step(control)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
            if a[2] or a[3]:
                break
        if task == "pendulum":
            assert _ + 1 == 200 and a[3] and not a[2]


def test_lander_reset_state_shaping_and_subsequent_native_step(monkeypatch):
    with pilot.environment("lunarlander", True) as env:
        observation, _ = env.reset(seed=0)
        base = env.unwrapped
        # The returned reset state reflects actual bodies, not an overwritten observation.
        assert observation[2] == np.float32(base.lander.linearVelocity.x * 10 / 50)
        assert observation[4] == np.float32(base.lander.angle)
        assert len(base.legs) == 2 and len(list(base.world.joints)) == 2
        shaping = (
            -100 * np.linalg.norm(observation[:2].astype(float))
            - 100 * np.linalg.norm(observation[2:4].astype(float))
            - 100 * abs(float(observation[4]))
            + 10 * sum(observation[6:])
        )
        assert base.prev_shaping == pytest.approx(shaping, abs=1e-4)
        # From this state, bypass the override: every later transition must still match.
        expected = env.step(2)
        env.reset(seed=0)
        monkeypatch.setattr(base, "step", pilot.LunarLander.step.__get__(base))
        actual = env.step(2)
        np.testing.assert_array_equal(expected[0], actual[0])
        assert expected[1:] == actual[1:]
