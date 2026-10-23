"""Pure tic-tac-toe rules and an exact minimax O policy."""

from functools import cache

LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def winner(board):
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None


def legal_moves(board):
    return [index for index, cell in enumerate(board) if cell == 0]


def play(board, move, player):
    cells = list(board)
    cells[move] = player
    return tuple(cells)


@cache
def _outcome(board, player):
    result = winner(board)
    if result is not None:
        return result
    values = (_outcome(play(board, move, player), -player) for move in legal_moves(board))
    return max(values) if player == 1 else min(values)


def minimax_move(board):
    moves = legal_moves(board)
    if not moves:
        return None
    return min(moves, key=lambda move: _outcome(play(board, move, -1), 1))
