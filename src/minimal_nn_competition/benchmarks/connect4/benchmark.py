"""Connect-4 benchmark: a 7-output net plays first against a 1-ply opponent."""

from __future__ import annotations

import statistics
from typing import Any

from minimal_nn_competition.benchmarks.connect4.engine import (
    legal_moves,
    opponent_move,
    play,
    winner,
)
from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    materialize_submission,
)

DEFAULT_SEEDS = tuple(range(20))
INPUT_SIZE = 42
OUTPUT_SIZE = 7


def _play_game(model: Any) -> float:
    import torch

    board = (0,) * 42
    while True:
        observation = torch.tensor(board, dtype=torch.float32)
        logits = model(observation.unsqueeze(0)).squeeze(0)
        move = max(legal_moves(board), key=lambda col: logits[col].item())
        board = play(board, move, 1)
        result = winner(board)
        if result is None:
            move = opponent_move(board)
            assert move is not None
            board = play(board, move, -1)
            result = winner(board)
        if result is not None:
            return 1.0 if result == 1 else 0.5 if result == 0 else 0.0


class Connect4Benchmark:
    name = "connect4"

    def __init__(self, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> None:
        if not seeds:
            raise ValueError("Connect-4 benchmark requires at least one seed")
        self.seeds = seeds

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Connect4")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("Connect4 submissions must declare input_size 42")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("Connect4 submissions must declare output_size 7")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
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
            all(score >= 0.5 for score in scores),
            {
                "scores": scores,
                "results": ["win" if s == 1.0 else "draw" if s == 0.5 else "loss" for s in scores],
                "seeds": len(self.seeds),
            },
        )
