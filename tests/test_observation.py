import pytest

from minimal_nn_competition.benchmarks.snake.engine import SnakeEngine
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.snake.observation import OBSERVATION_SIZE, observe


def _engine_with_body() -> SnakeEngine:
    engine = SnakeEngine(size=8, seed=0)
    engine.set_snake([(4, 4), (4, 5), (6, 4), (6, 6)])
    engine.food = (0, 0)
    return engine


def test_observation_has_24_features() -> None:
    engine = SnakeEngine(size=8, seed=0)
    values = observe(engine)
    assert len(values) == OBSERVATION_SIZE
    assert all(isinstance(value, float) for value in values)


def test_observation_selects_requested_features() -> None:
    engine = _engine_with_body()
    selected = observe(engine, (0, 1, 2, 4, 7))
    full = observe(engine)
    assert selected == (full[0], full[1], full[2], full[4], full[7])


def test_observation_encodes_head_food_and_body_distances() -> None:
    values = observe(_engine_with_body())
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(4 / 6)
    assert values[3] == pytest.approx(4 / 6)
    assert values[13] == pytest.approx(0.5)
    assert values[16] == pytest.approx(5 / 6)
    assert values[18] == pytest.approx(1.0)
    assert values[20] == pytest.approx(5 / 6)
    assert values[17] == 0.0


def test_environment_applies_input_selection() -> None:
    environment = SnakeEnvironment(size=8, seed=0, input_indices=(0, 1, 2, 4, 7))
    observation = environment.reset()
    assert len(observation) == 5
    assert observe(environment.engine, (0, 1, 2, 4, 7)) == observation
