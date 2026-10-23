"""Connect-4 benchmark: a 7-output net plays first against a 1-ply opponent."""

import statistics

from minimal_nn_competition.benchmarks.connect4.engine import (
    legal_moves,
    opponent_move,
    play,
    winner,
)
from minimal_nn_competition.model.entry import materialize_submission

DEFAULT_SEEDS = tuple(range(20))
INPUT_SIZE = 42
OUTPUT_SIZE = 7


def _play_game(model):
    import torch

    board = (0,) * INPUT_SIZE
    while True:
        observation = torch.tensor(board, dtype=torch.float32)
        logits = model(observation.unsqueeze(0)).squeeze(0)
        move = max(legal_moves(board), key=lambda col: logits[col].item())
        board = play(board, move, 1)
        result = winner(board)
        if result is None:
            board = play(board, opponent_move(board), -1)
            result = winner(board)
        if result is not None:
            return 0.5 * (1 + result)


class Connect4Benchmark:
    name = "connect4"

    def __init__(self, seeds=DEFAULT_SEEDS):
        if not seeds:
            raise ValueError("Connect-4 benchmark requires at least one seed")
        self.seeds = seeds

    def load_submission(self, submission):
        config = submission.config
        if config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Connect4")
        if config.input_size != INPUT_SIZE or config.output_size != OUTPUT_SIZE:
            raise ValueError("Connect4 needs input_size 42 and output_size 7")
        return materialize_submission(submission)

    def evaluate(self, submission):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Connect4 evaluation") from exc
        scores = []
        with torch.no_grad():
            for _ in self.seeds:
                scores.append(_play_game(submission.model))
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores)
        return (
            mean,
            std,
            min(scores) >= 0.5,
            {
                "scores": scores,
                "results": ["win" if s == 1.0 else "draw" if s == 0.5 else "loss" for s in scores],
                "seeds": len(self.seeds),
            },
        )
