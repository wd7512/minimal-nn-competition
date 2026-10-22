"""Load the benchmark ABC into our own benchmark with a the pass/fail criteria and score on minimum parameters"""

from __future__ import annotations

import statistics
from typing import Any

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    materialize_submission,
)

# Be an 8x8 grid and an agent passess if it can fill out the 8x8 grid. If it does not it gets a secondary score which allows it to be placed on the leaderboard.
# Therefore we should have {score = snake size (max 64), network_size = network size} and the ranking is score first and network size second.

DEFAULT_SEEDS = tuple(range(100))


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
        if submission.config.input_size != 7 or submission.config.output_size != 4:
            raise ValueError("Snake submissions must declare input_size 7 and output_size 4")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Snake evaluation") from exc
        lengths = []
        with torch.no_grad():
            for seed in self.seeds:
                environment = SnakeEnvironment(seed=seed, max_moves=self.max_moves)
                observation = environment.reset()
                done = False
                while not done:
                    values = submission.model(
                        torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    )
                    action = int(torch.argmax(values, dim=1).item())
                    observation, _, done = environment.step(action)
                lengths.append(len(environment.snake))
        score = statistics.fmean(lengths)
        score_std = statistics.pstdev(lengths)
        board_size = environment.size * environment.size
        return (
            score,
            score_std,
            all(length == board_size for length in lengths),
            {
                "seed_lengths": lengths,
                "max_length": board_size,
                "seeds": len(self.seeds),
            },
        )
