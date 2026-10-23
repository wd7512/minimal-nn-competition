"""Tic-tac-toe benchmark package."""

from minimal_nn_competition.benchmarks.tictactoe.benchmark import TicTacToeBenchmark
from minimal_nn_competition.benchmarks.tictactoe.engine import (
    legal_moves,
    minimax_move,
    play,
    winner,
)

__all__ = ["TicTacToeBenchmark", "legal_moves", "minimax_move", "play", "winner"]
