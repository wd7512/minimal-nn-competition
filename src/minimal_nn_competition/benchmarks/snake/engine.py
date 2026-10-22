"""Bitboard Snake engine with no observation encoding."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache

Position = tuple[int, int]
Direction = tuple[int, int]
DIRECTIONS: tuple[Direction, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Geometry:
    """Precomputed masks for a square board of a given size."""

    size: int
    cell_bits: tuple[int, ...]
    board_mask: int
    left_mask: int
    right_mask: int
    top_mask: int
    bottom_mask: int


@cache
def geometry(size: int) -> Geometry:
    """Return cached bitboard masks for a square board."""
    if size < 3:
        raise ValueError("Snake board size must be at least 3")
    cell_bits = tuple(1 << (y * size + x) for y in range(size) for x in range(size))
    left_mask = 0
    right_mask = 0
    for y in range(size):
        left_mask |= 1 << (y * size)
        right_mask |= 1 << (y * size + size - 1)
    top_mask = (1 << size) - 1
    bottom_mask = ((1 << size) - 1) << (size * (size - 1))
    board_mask = (1 << (size * size)) - 1
    return Geometry(
        size=size,
        cell_bits=cell_bits,
        board_mask=board_mask,
        left_mask=left_mask,
        right_mask=right_mask,
        top_mask=top_mask,
        bottom_mask=bottom_mask,
    )


class SnakeEngine:
    """Bitboard Snake simulation. ``step`` does not build observations."""

    def __init__(self, size: int = 8, seed: int = 0, max_moves: int = 512) -> None:
        if max_moves <= 0:
            raise ValueError("Snake max_moves must be positive")
        self.size = size
        self.seed = seed
        self.max_moves = max_moves
        self.board_size = size * size
        self._geometry = geometry(size)
        self.reset()

    def reset(self) -> None:
        self.random = random.Random(self.seed)
        center = self.size // 2
        self.snake: list[Position] = [(center, center)]
        self.direction: Direction = (1, 0)
        self.moves = 0
        self.head_cell = center * self.size + center
        self.head_bit = self._geometry.cell_bits[self.head_cell]
        self.body_mask = 0
        self._place_food()

    @property
    def food(self) -> Position:
        food_cell = self.food_bit.bit_length() - 1
        return food_cell % self.size, food_cell // self.size

    @food.setter
    def food(self, position: Position) -> None:
        x, y = position
        self.food_bit = self._geometry.cell_bits[y * self.size + x]

    def set_snake(self, snake: list[Position]) -> None:
        """Replace the snake body and rebuild bitboard state."""
        if not snake:
            raise ValueError("Snake body must contain at least the head")
        self.snake = list(snake)
        x, y = self.snake[0]
        self.head_cell = y * self.size + x
        self.head_bit = self._geometry.cell_bits[self.head_cell]
        body_mask = 0
        for x, y in self.snake[1:]:
            body_mask |= self._geometry.cell_bits[y * self.size + x]
        self.body_mask = body_mask

    def _place_food(self) -> None:
        free = self._geometry.board_mask & ~(self.head_bit | self.body_mask)
        if not free:
            self.food_bit = 0
            return
        choice = self.random.randrange(free.bit_count())
        while choice:
            free &= free - 1
            choice -= 1
        self.food_bit = free & -free

    def step(self, action: int) -> tuple[float, bool]:
        """Advance one move. Returns ``(reward, done)`` without an observation."""
        if not 0 <= action < len(DIRECTIONS):
            raise ValueError("Snake action must be an integer from 0 to 3")
        requested = DIRECTIONS[action]
        direction_x, direction_y = self.direction
        if requested != (-direction_x, -direction_y):
            self.direction = requested
            direction_x, direction_y = requested

        head = self.head_bit
        masks = self._geometry
        if direction_x == 1:
            if head & masks.right_mask:
                return -1.0, True
            next_bit = head << 1
        elif direction_x == -1:
            if head & masks.left_mask:
                return -1.0, True
            next_bit = head >> 1
        elif direction_y == 1:
            if head & masks.bottom_mask:
                return -1.0, True
            next_bit = head << self.size
        else:
            if head & masks.top_mask:
                return -1.0, True
            next_bit = head >> self.size

        self.moves += 1
        next_cell = next_bit.bit_length() - 1
        if next_bit == self.food_bit:
            old_head = self.head_bit
            self.head_bit = next_bit
            self.head_cell = next_cell
            self.body_mask |= old_head
            self.snake.insert(0, (next_cell % self.size, next_cell // self.size))
            if len(self.snake) == self.board_size:
                return 1.0, True
            self._place_food()
            return 1.0, False

        tail_x, tail_y = self.snake[-1]
        tail_bit = masks.cell_bits[tail_y * self.size + tail_x]
        body_after_move = (self.body_mask | self.head_bit) & ~tail_bit
        if next_bit & body_after_move:
            return -1.0, True

        self.head_bit = next_bit
        self.head_cell = next_cell
        self.body_mask = body_after_move
        self.snake.pop()
        self.snake.insert(0, (next_cell % self.size, next_cell // self.size))
        return 0.0, self.moves >= self.max_moves
