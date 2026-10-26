"""Fixed perfect-play agent used by the Snake Simple 8x8 environment speed benchmark."""

from minimal_nn_competition.benchmarks.common.snake.engine import DIRECTIONS

SIZE = 8
BOARD_SIZE = SIZE * SIZE
AGENT_NAME = "hamiltonian-cycle-v1"

HAMILTONIAN_CYCLE = tuple(
    zip(
        map(int, "4444456777777776666666555555544433322222221111111000000001233333"),
        map(int, "4321000012345677654321123456776556776543211234567765432100001234"),
        strict=True,
    )
)
_SUCCESSOR = {
    cell: HAMILTONIAN_CYCLE[(index + 1) % BOARD_SIZE]
    for index, cell in enumerate(HAMILTONIAN_CYCLE)
}
_ACTION_BY_DIRECTION = {direction: index for index, direction in enumerate(DIRECTIONS)}


def _build_actions():
    return tuple(
        _ACTION_BY_DIRECTION[(next_x - x, next_y - y)]
        for y in range(SIZE)
        for x in range(SIZE)
        for next_x, next_y in (_SUCCESSOR[(x, y)],)
    )


ACTIONS_BY_CELL = _build_actions()


def action_for(engine):
    return ACTIONS_BY_CELL[engine.head_cell]
