"""Small deterministic Snake simulation used by the benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass

Position = tuple[int, int]
Direction = tuple[int, int]
DIRECTIONS: tuple[Direction, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class SnakeEnvironment:
    size: int = 8
    seed: int = 0
    max_moves: int = 512

    def reset(self) -> tuple[int, ...]:
        self.random = random.Random(self.seed)
        center = self.size // 2
        self.snake: list[Position] = [(center, center)]
        self.direction = (1, 0)
        self.moves = 0
        self.food = self._new_food()
        return self.observation()

    def _new_food(self) -> Position:
        occupied = set(self.snake)
        available = [
            (x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied
        ]
        return self.random.choice(available)

    def observation(self) -> tuple[int, ...]:
        head = self.snake[0]
        return (head[0], head[1], self.food[0], self.food[1], *self.direction, len(self.snake))

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
        if not 0 <= action < len(DIRECTIONS):
            raise ValueError("Snake action must be an integer from 0 to 3")
        requested = DIRECTIONS[action]
        if requested != (-self.direction[0], -self.direction[1]):
            self.direction = requested
        head = self.snake[0]
        next_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.moves += 1
        tail = self.snake[-1]
        moves_into_tail = next_head == tail and next_head != self.food
        if (
            next_head[0] not in range(self.size)
            or next_head[1] not in range(self.size)
            or (next_head in self.snake and not moves_into_tail)
        ):
            return self.observation(), -1.0, True
        self.snake.insert(0, next_head)
        if next_head == self.food:
            if len(self.snake) == self.size * self.size:
                return self.observation(), 1.0, True
            self.food = self._new_food()
            reward = 1.0
        else:
            self.snake.pop()
            reward = 0.0
        return self.observation(), reward, self.moves >= self.max_moves
