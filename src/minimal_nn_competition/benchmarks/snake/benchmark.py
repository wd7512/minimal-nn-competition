"""Snake benchmark: pass/fail on filling the board, then minimum parameters."""

from __future__ import annotations

import statistics
from typing import Any

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.snake.observation import OBSERVATION_SIZE
from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    SubmissionConfig,
    materialize_submission,
)

# Be an 8x8 grid and an agent passes if it can fill out the 8x8 grid. If it does
# not it gets a secondary score which allows it to be placed on the leaderboard.
# Therefore we should have {score = snake size (max 64), network_size = network size}
# and the ranking is score first and network size second.

DEFAULT_SEEDS = tuple(range(100))


def selected_input_indices(config: SubmissionConfig) -> tuple[int, ...] | None:
    """Resolve the optional submission feature selection.

    ``None`` or ``true`` means all 24 features. A list selects those features in
    order, matching the model's first-layer input width.
    """
    indices = config.input_indices
    if indices is None or indices is True:
        return None
    if indices is False:
        raise ValueError("Snake input_indices must be true or a list of feature indices")
    if any(index >= OBSERVATION_SIZE for index in indices):
        raise ValueError(f"Snake input_indices must be in 0..{OBSERVATION_SIZE - 1}")
    return tuple(indices)


class SnakeBenchmark:
    name = "snake"

    def __init__(self, seeds: tuple[int, ...] = DEFAULT_SEEDS, max_moves: int = 4096) -> None:
        if not seeds:
            raise ValueError("Snake benchmark requires at least one seed")
        if max_moves <= 0:
            raise ValueError("Snake benchmark max_moves must be positive")
        self.seeds = seeds
        self.max_moves = max_moves

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Snake")
        indices = selected_input_indices(submission.config)
        expected_input = OBSERVATION_SIZE if indices is None else len(indices)
        if submission.config.input_size != expected_input:
            raise ValueError(
                f"Snake submissions must declare input_size {expected_input} "
                f"for the selected observation features"
            )
        if submission.config.output_size != 4:
            raise ValueError("Snake submissions must declare output_size 4")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Snake evaluation") from exc
        indices = selected_input_indices(submission.config)
        lengths = []
        moves = []
        with torch.no_grad():
            for seed in self.seeds:
                environment = SnakeEnvironment(
                    seed=seed,
                    max_moves=self.max_moves,
                    input_indices=indices,
                )
                observation = environment.reset()
                done = False
                ticks = 0
                while not done:
                    values = submission.model(
                        torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    )
                    action = int(torch.argmax(values, dim=1).item())
                    observation, _, done = environment.step(action)
                    ticks += 1
                lengths.append(len(environment.snake))
                moves.append(ticks)
        score = statistics.fmean(lengths)
        score_std = statistics.pstdev(lengths)
        board_size = environment.size * environment.size
        return (
            score,
            score_std,
            all(length == board_size for length in lengths),
            {
                "seed_lengths": lengths,
                "seed_moves": moves,
                "max_length": board_size,
                "seeds": len(self.seeds),
            },
        )
