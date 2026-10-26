"""Evaluation loop shared by the Snake benchmarks.

The per-size variants in :mod:`minimal_nn_competition.benchmarks.snake_simple_8x8`,
:mod:`minimal_nn_competition.benchmarks.snake_simple_16x16`, and
:mod:`minimal_nn_competition.benchmarks.snake_obstacles_8x8` only set the board
configuration; the pass/fail rule lives here.
"""

import statistics

from minimal_nn_competition.benchmarks.common.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.common.snake.observation import OBSERVATION_SIZE
from minimal_nn_competition.model.entry import materialize_submission

# Score is the fraction of seeds that fill the board (1.0 = every seed passes),
# so all passing entries tie and ranking falls through to the smaller network.
# Seed length and move count stay in details for the per-seed view.

DEFAULT_SEEDS = tuple(range(100))


def selected_input_indices(config, label=None):
    """Resolve the optional submission feature selection.

    ``None`` or ``true`` means all 24 features. A list selects those features in
    order, matching the model's first-layer input width.
    """
    label = label or config.benchmark
    indices = config.input_indices
    if indices is None or indices is True:
        return None
    if indices is False:
        raise ValueError(f"{label} input_indices must be true or a list of feature indices")
    if any(index >= OBSERVATION_SIZE for index in indices):
        raise ValueError(f"{label} input_indices must be in 0..{OBSERVATION_SIZE - 1}")
    return tuple(indices)


class BaseSnakeBenchmark:
    """Shared fill-the-board evaluation; subclasses set the board configuration."""

    name = ""
    size = 8
    loop_on_full = False
    # Per-food tick budget (None = no per-food limit). When set, a seed fails
    # if the snake goes this many ticks without growing: it stalls loops
    # without punishing slow-but-correct play.
    moves_per_food = None
    max_moves_default = 4096

    def __init__(self, seeds=DEFAULT_SEEDS, max_moves=None):
        label = self.name
        if not seeds:
            raise ValueError(f"{label} benchmark requires at least one seed")
        if max_moves is None:
            max_moves = self.max_moves_default
        if max_moves <= 0:
            raise ValueError(f"{label} benchmark max_moves must be positive")
        self.seeds = seeds
        self.max_moves = max_moves

    def obstacles(self, seed):
        """Return the blocked cells for a seed (none for plain Snake)."""
        return ()

    def environment_options(self, seed):
        """Return the size, blocked cells, and full-board rule for a seed."""
        return {
            "size": self.size,
            "blocked": self.obstacles(seed),
            "loop_on_full": self.loop_on_full,
        }

    def _environment(self, seed, input_indices):
        return SnakeEnvironment(
            seed=seed,
            max_moves=self.max_moves,
            input_indices=input_indices,
            **self.environment_options(seed),
        )

    def load_submission(self, submission):
        label = self.name
        if submission.config.benchmark != self.name:
            raise ValueError(f"Submission benchmark does not match {label}")
        indices = selected_input_indices(submission.config, label)
        expected_input = OBSERVATION_SIZE if indices is None else len(indices)
        if submission.config.input_size != expected_input:
            raise ValueError(
                f"{label} submissions must declare input_size {expected_input} "
                "for the selected observation features"
            )
        if submission.config.output_size != 4:
            raise ValueError(f"{label} submissions must declare output_size 4")
        return materialize_submission(submission)

    def evaluate(self, submission):
        label = self.name
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"PyTorch is required for {label} evaluation") from exc
        indices = selected_input_indices(submission.config, label)
        lengths = []
        moves = []
        seed_passed = []
        target_lengths = []
        seed_blocked = []
        with torch.no_grad():
            for seed in self.seeds:
                blocked = self.obstacles(seed)
                target = self.size * self.size - len(blocked)
                environment = self._environment(seed, indices)
                observation = environment.reset()
                done = False
                ticks = 0
                last_length = len(environment.snake)
                since_eat = 0
                while not done:
                    action = (
                        submission.model(torch.tensor(observation, dtype=torch.float32)[None])
                        .argmax(1)
                        .item()
                    )
                    observation, _, done = environment.step(action)
                    ticks += 1
                    if len(environment.snake) > last_length:
                        last_length = len(environment.snake)
                        since_eat = 0
                    else:
                        since_eat += 1
                        if self.moves_per_food is not None and since_eat > self.moves_per_food:
                            done = True
                length = len(environment.snake)
                won = bool(getattr(environment, "won", False))
                lengths.append(length)
                moves.append(ticks)
                target_lengths.append(target)
                seed_blocked.append([list(cell) for cell in blocked])
                seed_passed.append(
                    length == target
                    and ticks <= self.max_moves
                    and (won if self.loop_on_full else True)
                )
        score = statistics.fmean(seed_passed)
        return (
            score,
            all(seed_passed),
            {
                "seed_lengths": lengths,
                "seed_moves": moves,
                "seed_passed": seed_passed,
                "seed_target_lengths": target_lengths,
                "seed_blocked": seed_blocked,
            },
        )
