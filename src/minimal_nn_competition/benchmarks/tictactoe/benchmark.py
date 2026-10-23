"""Tic-tac-toe benchmark: a 9-output net plays X against an exact minimax O."""

from __future__ import annotations

import statistics
from typing import Any

from minimal_nn_competition.benchmarks.tictactoe.engine import (
    Board,
    legal_moves,
    minimax_move,
    play,
    winner,
)
from minimal_nn_competition.model.entry import (
    LoadedSubmission,
    Submission,
    materialize_submission,
)

DEFAULT_SEEDS = tuple(range(100))
INPUT_SIZE = 9
OUTPUT_SIZE = 9

_SCORES = {1.0: "win", 0.5: "draw", 0.0: "loss"}


def _play_game(model: Any) -> float:
    import torch

    board: Board = (0,) * 9
    while True:
        observation = torch.tensor([float(cell) for cell in board], dtype=torch.float32)
        logits = model(observation.unsqueeze(0)).squeeze(0)
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_moves(board)] = 0.0
        board = play(board, int(torch.argmax(logits + mask).item()), 1)
        result = winner(board)
        if result is None:
            move = minimax_move(board)
            assert move is not None
            board = play(board, move, -1)
            result = winner(board)
        if result is not None:
            if result == 1:
                return 1.0
            return 0.5 if result == 0 else 0.0


class TicTacToeBenchmark:
    name = "tictactoe"

    def __init__(self, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> None:
        if not seeds:
            raise ValueError("Tic-tac-toe benchmark requires at least one seed")
        self.seeds = seeds

    def load_submission(self, submission: Submission) -> LoadedSubmission:
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match TicTacToe")
        if submission.config.input_size != INPUT_SIZE:
            raise ValueError("TicTacToe submissions must declare input_size 9")
        if submission.config.output_size != OUTPUT_SIZE:
            raise ValueError("TicTacToe submissions must declare output_size 9")
        return materialize_submission(submission)

    def evaluate(self, submission: LoadedSubmission) -> tuple[float, float, bool, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for TicTacToe evaluation") from exc
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
                "results": [_SCORES[score] for score in scores],
                "seeds": len(self.seeds),
            },
        )
