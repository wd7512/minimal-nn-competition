"""24-feature Snake observation encoding, kept separate from the engine.

Feature layout:
  0-3   head row/column and distance to the bottom/right edges
  4-7   corner-adjacency averages derived from those four values
  8-11  food alignment on the same row or column
  12-15 food alignment on a diagonal
  16-23 nearest body or blocked cell: right, left, down, up, down-right,
        up-left, down-left, up-right

Non-zero values are normalized with ``(size - value - 1) / (size - 2)``.
"""

OBSERVATION_SIZE = 24

# Ray paths from a cell for the eight obstacle/food directions used by the
# encoding from the Snake-Learning project:
# right, left, down, up, down-right, up-left, down-left, up-right.
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


def _all_features(engine):
    size = engine.size
    head_x, head_y = engine.snake[0]
    head_cell = engine.head_cell

    values = [head_y, head_x, size - head_y - 1, size - head_x - 1]
    values += [(values[i] + values[-~i % 4] - 1) / 2 for i in range(4)]
    values += [0.0] * 16

    food = engine.food
    if food is not None:
        food_x, food_y = food
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

    target_mask = engine.body_mask | engine.blocked_mask
    if target_mask:
        y, x = divmod(head_cell, size)
        for index, (dy, dx) in enumerate(_RAY_DIRECTIONS):
            next_y, next_x = y + dy, x + dx
            distance = 0
            while 0 <= next_y < size and 0 <= next_x < size:
                distance += 1
                if target_mask >> (next_y * size + next_x) & 1:
                    values[16 + index] = distance
                    break
                next_y += dy
                next_x += dx

    return tuple((size - value - 1) / (size - 2) if value else value for value in values)


def observe(engine, indices=None):
    """Return the full 24-feature observation or a selected subset."""
    values = _all_features(engine)
    if indices is None:
        return values
    return tuple(values[index] for index in indices)
