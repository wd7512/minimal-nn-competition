"""Deterministic obstacle layouts with verified closed Hamiltonian tours.

Obstacle positions are sampled as one or two random opposite-colour pairs so
that the free board keeps equal checkerboard colours. Every candidate is still
checked by :func:`find_closed_tour`; layouts without a found tour are discarded
and resampled from the same RNG stream.
"""

import random
from functools import cache

from minimal_nn_competition.benchmarks.snake.engine import DIRECTIONS, SnakeEngine, geometry

TOUR_NODE_BUDGET = 50_000
MAX_LAYOUT_ATTEMPTS = 1000


class _TourBudgetExceeded(Exception):
    pass


@cache
def _adjacency(size):
    cells = []
    for cell in range(size * size):
        y, x = divmod(cell, size)
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            next_y, next_x = y + dy, x + dx
            if 0 <= next_y < size and 0 <= next_x < size:
                neighbours.append(next_y * size + next_x)
        cells.append(tuple(neighbours))
    return tuple(cells)


@cache
def _adjacency_masks(size):
    return tuple(sum(1 << neighbour for neighbour in cells) for cells in _adjacency(size))


@cache
def _colour_masks(size):
    black = white = 0
    for cell in range(size * size):
        y, x = divmod(cell, size)
        if (x + y) % 2 == 0:
            black |= 1 << cell
        else:
            white |= 1 << cell
    return black, white


def _expand(frontier, unvisited, size):
    masks = geometry(size)
    board = masks.board_mask
    expanded = (
        ((frontier & ~masks.right_mask) << 1)
        | ((frontier & ~masks.left_mask) >> 1)
        | (frontier << size)
        | (frontier >> size)
    )
    return expanded & board & unvisited


def find_closed_tour(size, blocked):
    """Return a closed Hamiltonian tour over the unblocked cells, or ``None``."""
    normalised = tuple(sorted(tuple(cell) for cell in blocked))
    return _find_closed_tour(size, normalised)


@cache
def _find_closed_tour(size, blocked):
    """Cached closed-tour search for a normalised blocked-cell tuple.

    The tour starts at the centre, visits every free cell exactly once, ends on
    a cell adjacent to the centre, and its first move is not left.
    """
    blocked_mask = 0
    for x, y in blocked:
        if not (0 <= x < size and 0 <= y < size):
            return None
        blocked_mask |= 1 << (y * size + x)
    board = geometry(size).board_mask
    free = board & ~blocked_mask
    count = free.bit_count()
    if count % 2:
        return None
    start = (size // 2) * size + size // 2
    start_bit = 1 << start
    if not free & start_bit:
        return None
    black, white = _colour_masks(size)
    if (free & black).bit_count() != (free & white).bit_count():
        return None
    adjacency = _adjacency_masks(size)
    remaining_cells = free
    while remaining_cells:
        bit = remaining_cells & -remaining_cells
        remaining_cells -= bit
        cell = bit.bit_length() - 1
        if (adjacency[cell] & free).bit_count() < 2:
            return None

    nodes = 0
    path = [start]

    def search(current, unvisited, remaining):
        nonlocal nodes
        nodes += 1
        if nodes > TOUR_NODE_BUDGET:
            raise _TourBudgetExceeded
        if remaining == 0:
            return bool(adjacency[current] & start_bit)

        first = unvisited & -unvisited
        reached = first
        frontier = first
        while frontier:
            frontier = _expand(frontier, unvisited, size) & ~reached
            reached |= frontier
        if reached != unvisited:
            return False

        unvisited_count = unvisited.bit_count()
        colour = (current % size + current // size) % 2
        same = unvisited & (black if colour == 0 else white)
        opposite = unvisited & (white if colour == 0 else black)
        if (
            same.bit_count() != unvisited_count // 2
            or opposite.bit_count() != (unvisited_count + 1) // 2
        ):
            return False

        neighbours = adjacency[current] & unvisited
        candidates = []
        scanning = neighbours
        while scanning:
            bit = scanning & -scanning
            scanning -= bit
            cell = bit.bit_length() - 1
            if remaining == 1 and not adjacency[cell] & start_bit:
                continue
            onward = (adjacency[cell] & unvisited).bit_count()
            candidates.append((onward, cell))
        candidates.sort()

        for _, cell in candidates:
            next_unvisited = unvisited & ~(1 << cell)
            next_current_bit = 1 << cell
            blocked_degree = False
            checking = adjacency[current] & next_unvisited
            while checking:
                bit = checking & -checking
                checking -= bit
                neighbour = bit.bit_length() - 1
                available = (adjacency[neighbour] & (next_unvisited | next_current_bit)).bit_count()
                if adjacency[neighbour] & start_bit:
                    available += 1
                if available < 2:
                    blocked_degree = True
                    break
            if blocked_degree:
                continue
            path.append(cell)
            if search(cell, next_unvisited, remaining - 1):
                return True
            path.pop()
        return False

    try:
        found = search(start, free & ~start_bit, count - 1)
    except _TourBudgetExceeded:
        return None
    if not found:
        return None

    tour = list(path)
    start_y, start_x = divmod(tour[0], size)
    next_y, next_x = divmod(tour[1], size)
    if (next_x, next_y) == (start_x - 1, start_y):
        tour = [tour[0], *reversed(tour[1:])]
    return tuple((cell % size, cell // size) for cell in tour)


def _sample_obstacles(rng, size, count):
    start = (size // 2, size // 2)
    cells = [(x, y) for y in range(size) for x in range(size) if (x, y) != start]
    black = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 0]
    white = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 1]
    chosen = []
    for _ in range(count // 2):
        first = rng.choice([cell for cell in black if cell not in chosen])
        second = rng.choice([cell for cell in white if cell not in chosen])
        chosen += [first, second]
    return tuple(sorted(chosen))


@cache
def obstacles_for_seed(size, seed):
    """Return the deterministic blocked cells chosen for ``seed``."""
    rng = random.Random(seed)
    count = rng.choice((2, 4))
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        blocked = _sample_obstacles(rng, size, count)
        if find_closed_tour(size, blocked) is not None:
            return blocked
    raise RuntimeError(f"Could not find a passable obstacle layout for seed {seed}")


def _adjacent(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


def _check_tour(size, blocked, tour):
    start = (size // 2, size // 2)
    free = {(x, y) for y in range(size) for x in range(size)} - set(blocked)
    if tour[0] != start:
        raise AssertionError("Closed tour does not start at the centre")
    if len(tour) != len(free) or len(set(tour)) != len(tour) or set(tour) != free:
        raise AssertionError("Closed tour does not visit every free cell exactly once")
    if any(not _adjacent(first, second) for first, second in zip(tour, tour[1:], strict=False)):
        raise AssertionError("Closed tour has a non-adjacent step")
    if not _adjacent(tour[-1], start):
        raise AssertionError("Closed tour does not end next to the centre")
    if tour[1] == (start[0] - 1, start[1]):
        raise AssertionError("Closed tour starts by moving left")


def _action_between(first, second):
    return DIRECTIONS.index((second[0] - first[0], second[1] - first[1]))


def _follow_tour(engine, tour):
    positions = {cell: index for index, cell in enumerate(tour)}
    moves = 0
    while not engine.won:
        if moves >= engine.max_moves:
            break
        current = engine.snake[0]
        next_cell = tour[(positions[current] + 1) % len(tour)]
        _, done = engine.step(_action_between(current, next_cell))
        moves += 1
        if done and not engine.won:
            break
    return moves


def validate_seed(size, seed, max_moves=4096):
    """Re-check the layout's tour and prove a tour follower completes the game."""
    blocked = obstacles_for_seed(size, seed)
    start = (size // 2, size // 2)
    if len(blocked) not in (2, 4):
        raise AssertionError(f"Seed {seed} has {len(blocked)} blocked cells, expected 2 or 4")
    if start in blocked:
        raise AssertionError(f"Seed {seed} blocks the centre start cell")
    blocked_colours = [(x + y) % 2 for x, y in blocked]
    if blocked_colours.count(0) != blocked_colours.count(1):
        raise AssertionError(f"Seed {seed} does not block equal numbers of each colour")
    tour = find_closed_tour(size, blocked)
    if tour is None:
        raise AssertionError(f"No closed tour found for seed {seed}")
    _check_tour(size, blocked, tour)

    engine = SnakeEngine(
        size=size,
        seed=seed,
        max_moves=max_moves,
        blocked=blocked,
        loop_on_full=True,
    )
    engine.reset()
    moves = _follow_tour(engine, tour)
    free_count = size * size - len(blocked)
    if len(engine.snake) != free_count:
        raise AssertionError(
            f"Seed {seed} reached length {len(engine.snake)}, expected {free_count}"
        )
    if not engine.won:
        raise AssertionError(f"Seed {seed} did not complete the extra loop move")
    if moves > max_moves:
        raise AssertionError(f"Seed {seed} took {moves} moves, limit is {max_moves}")
    return {
        "seed": seed,
        "blocked": blocked,
        "tour_length": len(tour),
        "moves": moves,
        "length": len(engine.snake),
    }
