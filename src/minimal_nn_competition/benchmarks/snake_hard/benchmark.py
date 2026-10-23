"""Snake-hard benchmark: same Snake engine, board size shifts per seed."""

import statistics

from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark, selected_input_indices
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

DEFAULT_SEEDS = tuple(range(20))
BOARD_SIZES = 6, 8, 10


def board_size_for_seed(seed):
    """Cycle board sizes 6/8/10 by seed; that cycle is the whole shift."""
    return BOARD_SIZES[seed % len(BOARD_SIZES)]


class SnakeHardBenchmark(SnakeBenchmark):
    name = "snake-hard"

    def __init__(self, seeds=DEFAULT_SEEDS, max_moves=4096):
        super().__init__(seeds, max_moves)

    def evaluate(self, submission):
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
            lengths == [size * size for size in sizes],
            {
                "seed_lengths": lengths,
                "seed_moves": moves,
                "board_sizes": sizes,
                "seeds": len(self.seeds),
            },
        )
