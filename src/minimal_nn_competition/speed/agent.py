"""Fixed perfect-play agent used by the Snake environment speed benchmark."""

from __future__ import annotations

from minimal_nn_competition.benchmarks.snake.environment import DIRECTIONS

SIZE = 8
BOARD_SIZE = SIZE * SIZE
AGENT_NAME = "hamiltonian-cycle-v1"

HAMILTONIAN_CYCLE: tuple[tuple[int, int], ...] = (
    (4, 4),
    (4, 3),
    (4, 2),
    (4, 1),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 5),
    (7, 6),
    (7, 7),
    (6, 7),
    (6, 6),
    (6, 5),
    (6, 4),
    (6, 3),
    (6, 2),
    (6, 1),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (4, 7),
    (4, 6),
    (4, 5),
    (3, 5),
    (3, 6),
    (3, 7),
    (2, 7),
    (2, 6),
    (2, 5),
    (2, 4),
    (2, 3),
    (2, 2),
    (2, 1),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (0, 7),
    (0, 6),
    (0, 5),
    (0, 4),
    (0, 3),
    (0, 2),
    (0, 1),
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
)

_SUCCESSOR = {
    cell: HAMILTONIAN_CYCLE[(index + 1) % BOARD_SIZE]
    for index, cell in enumerate(HAMILTONIAN_CYCLE)
}
_ACTION_BY_DIRECTION = {direction: index for index, direction in enumerate(DIRECTIONS)}


def _build_actions() -> tuple[int, ...]:
    actions = []
    for y in range(SIZE):
        for x in range(SIZE):
            next_x, next_y = _SUCCESSOR[(x, y)]
            actions.append(_ACTION_BY_DIRECTION[(next_x - x, next_y - y)])
    return tuple(actions)


ACTIONS_BY_CELL = _build_actions()


def action_for(observation: tuple[int, ...]) -> int:
    return ACTIONS_BY_CELL[observation[1] * SIZE + observation[0]]
