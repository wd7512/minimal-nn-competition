"""Compatibility wrapper combining the Snake engine and observation encoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from minimal_nn_competition.benchmarks.snake.engine import (
    DIRECTIONS,
    Direction,
    Position,
    SnakeEngine,
)
from minimal_nn_competition.benchmarks.snake.observation import observe

__all__ = ["DIRECTIONS", "SnakeEngine", "SnakeEnvironment"]


class SnakeEnvironment:
    """Engine plus observation wrapper used by evaluation and training."""

    def __init__(
        self,
        size: int = 8,
        seed: int = 0,
        max_moves: int = 512,
        input_indices: Sequence[int] | None = None,
    ) -> None:
        self.engine = SnakeEngine(size=size, seed=seed, max_moves=max_moves)
        self.input_indices = tuple(input_indices) if input_indices is not None else None

    @property
    def snake(self) -> list[Position]:
        return self.engine.snake

    @snake.setter
    def snake(self, snake: list[Position]) -> None:
        self.engine.set_snake(snake)

    @property
    def food(self) -> Position:
        return self.engine.food

    @food.setter
    def food(self, position: Position) -> None:
        self.engine.food = position

    @property
    def direction(self) -> Direction:
        return self.engine.direction

    @direction.setter
    def direction(self, direction: Direction) -> None:
        self.engine.direction = direction

    def __getattr__(self, name: str) -> Any:
        engine = self.__dict__.get("engine")
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)

    def reset(self) -> tuple[float, ...]:
        self.engine.reset()
        return self.observation()

    def observation(self) -> tuple[float, ...]:
        return observe(self.engine, self.input_indices)

    def advance(self, action: int) -> tuple[float, bool]:
        """Advance the engine without building an observation."""
        return self.engine.step(action)

    def step(self, action: int) -> tuple[tuple[float, ...], float, bool]:
        reward, done = self.engine.step(action)
        return self.observation(), reward, done
