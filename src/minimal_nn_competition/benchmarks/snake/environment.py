"""Compatibility wrapper combining the Snake engine and observation encoding."""

from minimal_nn_competition.benchmarks.snake.engine import DIRECTIONS, SnakeEngine
from minimal_nn_competition.benchmarks.snake.observation import observe

__all__ = ["DIRECTIONS", "SnakeEngine", "SnakeEnvironment"]


class SnakeEnvironment:
    """Engine plus observation wrapper used by evaluation and training."""

    def __init__(
        self,
        size=8,
        seed=0,
        max_moves=512,
        input_indices=None,
        blocked=(),
        loop_on_full=False,
    ):
        self.engine = SnakeEngine(
            size=size,
            seed=seed,
            max_moves=max_moves,
            blocked=blocked,
            loop_on_full=loop_on_full,
        )
        self.input_indices = tuple(input_indices) if input_indices is not None else None

    @property
    def snake(self):
        return self.engine.snake

    @snake.setter
    def snake(self, snake):
        self.engine.set_snake(snake)

    @property
    def food(self):
        return self.engine.food

    @food.setter
    def food(self, position):
        self.engine.food = position

    @property
    def direction(self):
        return self.engine.direction

    @direction.setter
    def direction(self, direction):
        self.engine.direction = direction

    def __getattr__(self, name):
        engine = self.__dict__.get("engine")
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)

    def reset(self):
        self.engine.reset()
        return self.observation()

    def observation(self):
        return observe(self.engine, self.input_indices)

    def advance(self, action):
        """Advance the engine without building an observation."""
        return self.engine.step(action)

    def step(self, action):
        reward, done = self.engine.step(action)
        return self.observation(), reward, done
