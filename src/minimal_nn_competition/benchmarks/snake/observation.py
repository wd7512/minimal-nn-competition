"""24-feature Snake observation encoding, kept separate from the engine.

Feature layout:
  0-3   head row/column and distance to the bottom/right edges
  4-7   corner-adjacency averages derived from those four values
  8-11  food alignment on the same row or column
  12-15 food alignment on a diagonal
  16-23 nearest body segment: right, left, down, up, down-right, up-left,
        down-left, up-right

Non-zero values are normalized with ``(size - value - 1) / (size - 2)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_nn_competition.benchmarks.snake.engine import SnakeEngine

OBSERVATION_SIZE = 24

# Ray paths from a cell for the eight obstacle/food directions used by the
# encoding from the Snake-Learning project:
# right, left, down, up, down-right, up-left, down-left, up-right.
_RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)


@cache
def _rays(size: int) -> tuple[tuple[int, ...], ...]:
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


def _nearest_low(mask: int, ray: int) -> int | None:
    candidates = mask & ray
    if not candidates:
        return None
    return (candidates & -candidates).bit_length() - 1


def _nearest_high(mask: int, ray: int) -> int | None:
    candidates = mask & ray
    if not candidates:
        return None
    return candidates.bit_length() - 1


def _all_features(engine: SnakeEngine) -> tuple[float, ...]:
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
        rays = _rays(size)[head_cell]
        right, left, down, up, down_right, up_left, down_left, up_right = rays

        position = _nearest_low(body_mask, right)
        if position is not None:
            values[16] = position - head_cell
        position = _nearest_high(body_mask, left)
        if position is not None:
            values[17] = head_cell - position
        position = _nearest_low(body_mask, down)
        if position is not None:
            values[18] = (position - head_cell) // size
        position = _nearest_high(body_mask, up)
        if position is not None:
            values[19] = (head_cell - position) // size
        position = _nearest_low(body_mask, down_right)
        if position is not None:
            values[20] = (position - head_cell) // (size + 1)
        position = _nearest_high(body_mask, up_left)
        if position is not None:
            values[21] = (head_cell - position) // (size + 1)
        position = _nearest_low(body_mask, down_left)
        if position is not None:
            values[22] = (position - head_cell) // (size - 1)
        position = _nearest_high(body_mask, up_right)
        if position is not None:
            values[23] = (head_cell - position) // (size - 1)

    denominator = size - 2
    for index, value in enumerate(values):
        if value:
            values[index] = (size - value - 1) / denominator
    return tuple(values)


def observe(engine: SnakeEngine, indices: Sequence[int] | None = None) -> tuple[float, ...]:
    """Return the full 24-feature observation or a selected subset."""
    values = _all_features(engine)
    if indices is None:
        return values
    return tuple(values[index] for index in indices)
