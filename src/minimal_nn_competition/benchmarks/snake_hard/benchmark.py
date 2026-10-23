"""Snake-hard benchmark: same Snake engine, board size shifts per seed."""

from __future__ import annotations

import statistics
from typing import Any

from minimal_nn_competition.benchmarks.snake.benchmark import (
    SnakeBenchmark,
    selected_input_indices,
)
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.model.entry import LoadedSubmission

DEFAULT_SEEDS = tuple(range(20))
BOARD_SIZES = (6, 8, 10)


def board_size_for_seed(seed: int) -> int:
    """Cycle board sizes 6/8/10 by seed; that cycle is the whole shift."""
    return BOARD_SIZES[seed % len(BOARD_SIZES)]


class SnakeHardBenchmark(SnakeBenchmark):
    name = "snake-hard"

    def __init__(self, seeds: tuple[int, ...] = DEFAULT_SEEDS, max_moves: int = 4096) -> None:
        super().__init__(seeds=seeds, max_moves=max_moves)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Snake evaluation") from exc
        indices = selected_input_indices(submission.config)
        lengths = []
        moves = []
        sizes = []
        with torch.no_grad():
            for seed in self.seeds:
                size = board_size_for_seed(seed)
                sizes.append(size)
                environment = SnakeEnvironment(
                    size=size,
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
        return (
            statistics.fmean(lengths),
            statistics.pstdev(lengths),
            all(length == size * size for length, size in zip(lengths, sizes, strict=True)),
            {
                "seed_lengths": lengths,
                "seed_moves": moves,
                "board_sizes": sizes,
                "seeds": len(self.seeds),
            },
        )
