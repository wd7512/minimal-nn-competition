"""Bitboard Snake engine with no observation encoding."""

import random
from dataclasses import dataclass
from functools import cache

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


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
def geometry(size):
    """Return cached bitboard masks for a square board."""
    if size < 3:
        raise ValueError("Snake board size must be at least 3")
    top_mask = (1 << size) - 1
    board_mask = (1 << (size * size)) - 1
    left_mask = board_mask // top_mask
    right_mask = left_mask << (size - 1)
    bottom_mask = top_mask << (size * (size - 1))
    cell_bits = tuple(1 << cell for cell in range(size * size))
    return Geometry(size, cell_bits, board_mask, left_mask, right_mask, top_mask, bottom_mask)


class SnakeEngine:
    """Bitboard Snake simulation. ``step`` does not build observations."""

    def __init__(self, size=8, seed=0, max_moves=512):
        if max_moves <= 0:
            raise ValueError("Snake max_moves must be positive")
        self.size = size
        self.seed = seed
        self.max_moves = max_moves
        self.board_size = size * size
        self._geometry = geometry(size)
        self.reset()

    def reset(self):
        self.random = random.Random(self.seed)
        center = self.size // 2
        self.snake = [(center, center)]
        self.direction = (1, 0)
        self.moves = 0
        self.head_cell = center * self.size + center
        self.head_bit = self._geometry.cell_bits[self.head_cell]
        self.body_mask = 0
        self._place_food()

    @property
    def food(self):
        food_cell = self.food_bit.bit_length() - 1
        return food_cell % self.size, food_cell // self.size

    @food.setter
    def food(self, position):
        x, y = position
        self.food_bit = self._geometry.cell_bits[y * self.size + x]

    def set_snake(self, snake):
        """Replace the snake body and rebuild bitboard state."""
        if not snake:
            raise ValueError("Snake body must contain at least the head")
        self.snake = list(snake)
        x, y = self.snake[0]
        self.head_cell = y * self.size + x
        cell_bits = self._geometry.cell_bits
        self.head_bit = cell_bits[self.head_cell]
        body_mask = 0
        for x, y in self.snake[1:]:
            body_mask |= cell_bits[y * self.size + x]
        self.body_mask = body_mask

    def _place_food(self):
        free = self._geometry.board_mask & ~(self.head_bit | self.body_mask)
        if not free:
            self.food_bit = 0
            return
        choice = self.random.randrange(free.bit_count())
        while choice:
            free &= free - 1
            choice -= 1
        self.food_bit = free & -free

    def step(self, action):
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
            edge, shift = masks.right_mask, 1
        elif direction_x == -1:
            edge, shift = masks.left_mask, -1
        elif direction_y == 1:
            edge, shift = masks.bottom_mask, self.size
        else:
            edge, shift = masks.top_mask, -self.size
        if head & edge:
            return -1.0, True
        next_bit = head << shift if shift > 0 else head >> -shift

        self.moves += 1
        next_cell = next_bit.bit_length() - 1
        new_head = (next_cell % self.size, next_cell // self.size)
        if next_bit == self.food_bit:
            old_head = self.head_bit
            self.head_bit = next_bit
            self.head_cell = next_cell
            self.body_mask |= old_head
            self.snake.insert(0, new_head)
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
        self.snake.insert(0, new_head)
        return 0.0, self.moves >= self.max_moves
