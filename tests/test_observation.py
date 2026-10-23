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


_RAY_DIRECTIONS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)


def _reference_rays(size: int) -> tuple[tuple[int, ...], ...]:
    rays = []
    for cell in range(size * size):
        y, x = divmod(cell, size)
        cell_rays = []
        for dy, dx in _RAY_DIRECTIONS:
            mask = 0
            next_y, next_x = y + dy, x + dx
            while 0 <= next_y < size and 0 <= next_x < size:
                mask |= 1 << (next_y * size + next_x)
                next_y += dy
                next_x += dx
            cell_rays.append(mask)
        rays.append(tuple(cell_rays))
    return tuple(rays)


def _reference_nearest(mask: int, ray: int, high: bool) -> int | None:
    candidates = mask & ray
    if not candidates:
        return None
    if high:
        return candidates.bit_length() - 1
    return (candidates & -candidates).bit_length() - 1


def _reference_observation(engine: SnakeEngine) -> tuple[float, ...]:
    size = engine.size
    head_x, head_y = engine.snake[0]
    food_x, food_y = engine.food
    head_cell = engine.head_cell

    values: list[float] = [0.0] * OBSERVATION_SIZE
    values[0] = head_y
    values[1] = head_x
    values[2] = size - head_y - 1
    values[3] = size - head_x - 1
    values[4] = (values[0] + values[1] - 1) / 2
    values[5] = (values[1] + values[2] - 1) / 2
    values[6] = (values[2] + values[3] - 1) / 2
    values[7] = (values[3] + values[0] - 1) / 2

    row_delta = head_y - food_y
    column_delta = head_x - food_x
    if row_delta == 0:
        values[8 if head_x > food_x else 9] = abs(column_delta)
    if column_delta == 0:
        values[10 if head_y > food_y else 11] = abs(row_delta)
    if row_delta == column_delta:
        values[12 if row_delta < 0 else 13] = abs(row_delta)
    if row_delta == -column_delta:
        values[14 if row_delta < 0 else 15] = abs(row_delta)

    body_mask = engine.body_mask
    if body_mask:
        rays = _reference_rays(size)[head_cell]
        right, left, down, up, down_right, up_left, down_left, up_right = rays

        position = _reference_nearest(body_mask, right, False)
        if position is not None:
            values[16] = position - head_cell
        position = _reference_nearest(body_mask, left, True)
        if position is not None:
            values[17] = head_cell - position
        position = _reference_nearest(body_mask, down, False)
        if position is not None:
            values[18] = (position - head_cell) // size
        position = _reference_nearest(body_mask, up, True)
        if position is not None:
            values[19] = (head_cell - position) // size
        position = _reference_nearest(body_mask, down_right, False)
        if position is not None:
            values[20] = (position - head_cell) // (size + 1)
        position = _reference_nearest(body_mask, up_left, True)
        if position is not None:
            values[21] = (head_cell - position) // (size + 1)
        position = _reference_nearest(body_mask, down_left, False)
        if position is not None:
            values[22] = (position - head_cell) // (size - 1)
        position = _reference_nearest(body_mask, up_right, True)
        if position is not None:
            values[23] = (head_cell - position) // (size - 1)

    denominator = size - 2
    for index, value in enumerate(values):
        if value:
            values[index] = (size - value - 1) / denominator
    return tuple(values)


def test_observation_matches_literal_reference_on_random_states() -> None:
    import random

    rng = random.Random(2024)
    for size in (3, 4, 5, 8):
        cells = [(x, y) for y in range(size) for x in range(size)]
        for _ in range(15):
            engine = SnakeEngine(size=size, seed=rng.randrange(1000))
            shuffled = cells[:]
            rng.shuffle(shuffled)
            engine.set_snake(shuffled[: rng.randrange(1, size * size + 1)])
            engine.food = (rng.randrange(size), rng.randrange(size))
            assert observe(engine) == _reference_observation(engine)


@pytest.mark.parametrize("size", [3, 4, 5, 8])
def test_geometry_masks_match_literal_definitions(size: int) -> None:
    from minimal_nn_competition.benchmarks.snake.engine import geometry

    masks = geometry(size)
    left_mask = 0
    right_mask = 0
    for y in range(size):
        left_mask |= 1 << (y * size)
        right_mask |= 1 << (y * size + size - 1)
    top_mask = (1 << size) - 1
    assert masks.left_mask == left_mask
    assert masks.right_mask == right_mask
    assert masks.top_mask == top_mask
    assert masks.bottom_mask == top_mask << (size * (size - 1))
    assert masks.board_mask == (1 << (size * size)) - 1
    assert masks.cell_bits == tuple(1 << (y * size + x) for y in range(size) for x in range(size))


_GOLDEN_SCENARIOS = (
    (
        4,
        7,
        [(1, 1), (1, 0), (0, 0)],
        (2, 1),
        (1, 0),
        [1, 2, 3, 0, 1, 2],
        [
            ((1.0, False), [(2, 1), (1, 1), (1, 0), (0, 0)], (0, 1), (1, 0), 1, 6, 35),
            ((0.0, False), [(2, 2), (2, 1), (1, 1), (1, 0)], (0, 1), (0, 1), 2, 10, 98),
            ((0.0, False), [(1, 2), (2, 2), (2, 1), (1, 1)], (0, 1), (-1, 0), 3, 9, 1120),
            ((0.0, False), [(1, 1), (1, 2), (2, 2), (2, 1)], (0, 1), (0, -1), 4, 5, 1600),
            ((0.0, False), [(2, 1), (1, 1), (1, 2), (2, 2)], (0, 1), (1, 0), 5, 6, 1568),
            ((0.0, False), [(2, 2), (2, 1), (1, 1), (1, 2)], (0, 1), (0, 1), 6, 10, 608),
        ],
    ),
    (
        5,
        3,
        [(2, 2), (2, 3), (1, 3)],
        (2, 0),
        (0, -1),
        [3, 3, 3, 0, 1, 2, 3],
        [
            ((0.0, False), [(1, 2), (2, 2), (2, 3)], (2, 0), (-1, 0), 1, 11, 135168),
            ((0.0, False), [(0, 2), (1, 2), (2, 2)], (2, 0), (-1, 0), 2, 10, 6144),
            ((-1.0, True), [(0, 2), (1, 2), (2, 2)], (2, 0), (-1, 0), 2, 10, 6144),
            ((0.0, False), [(0, 1), (0, 2), (1, 2)], (2, 0), (0, -1), 3, 5, 3072),
            ((0.0, False), [(1, 1), (0, 1), (0, 2)], (2, 0), (1, 0), 4, 6, 1056),
            ((0.0, False), [(1, 2), (1, 1), (0, 1)], (2, 0), (0, 1), 5, 11, 96),
            ((0.0, False), [(0, 2), (1, 2), (1, 1)], (2, 0), (-1, 0), 6, 10, 2112),
        ],
    ),
)


@pytest.mark.parametrize(
    ("size", "seed", "snake", "food", "direction", "actions", "expected"), _GOLDEN_SCENARIOS
)
def test_engine_step_golden_sequence(
    size: int,
    seed: int,
    snake: list[tuple[int, int]],
    food: tuple[int, int],
    direction: tuple[int, int],
    actions: list[int],
    expected: list[tuple[object, ...]],
) -> None:
    engine = SnakeEngine(size=size, seed=seed)
    engine.set_snake(snake)
    engine.food = food
    engine.direction = direction
    for action, row in zip(actions, expected, strict=True):
        assert engine.step(action) == row[0]
        assert engine.snake == row[1]
        assert engine.food == row[2]
        assert engine.direction == row[3]
        assert engine.moves == row[4]
        assert engine.head_cell == row[5]
        assert engine.body_mask == row[6]
