"""Pure connect-4 rules and a 1-ply win-block-centre opponent policy."""

from __future__ import annotations

Board = tuple[int, ...]

ROWS = 6
COLS = 7

_CENTRE_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _lines() -> tuple[tuple[int, int, int, int], ...]:
    lines = []
    for row in range(ROWS):
        for col in range(COLS - 3):
            base = row * COLS + col
            lines.append((base, base + 1, base + 2, base + 3))
    for row in range(ROWS - 3):
        for col in range(COLS):
            base = row * COLS + col
            lines.append((base, base + COLS, base + 2 * COLS, base + 3 * COLS))
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            base = row * COLS + col
            lines.append((base, base + COLS + 1, base + 2 * COLS + 2, base + 3 * COLS + 3))
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            base = row * COLS + col
            lines.append((base, base - COLS + 1, base - 2 * COLS + 2, base - 3 * COLS + 3))
    return tuple(lines)


LINES = _lines()


def winner(board: Board) -> int | None:
    """Return 1 (net wins), -1 (opponent wins), 0 (full-board draw), or None."""
    for a, b, c, d in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c] == board[d]:
            return board[a]
    if all(cell != 0 for cell in board):
        return 0
    return None


def legal_moves(board: Board) -> list[int]:
    """Return the non-full columns in ascending order."""
    return [col for col in range(COLS) if board[col] == 0]


def play(board: Board, col: int, player: int) -> Board:
    """Drop ``player``'s disc into ``col``; the disc lands on the lowest empty row."""
    cells = list(board)
    for row in range(ROWS - 1, -1, -1):
        index = row * COLS + col
        if cells[index] == 0:
            cells[index] = player
            return tuple(cells)
    raise ValueError(f"Column {col} is full")


def opponent_move(board: Board) -> int | None:
    """Win if possible, else block the net's immediate win, else centre-first pick."""
    moves = legal_moves(board)
    if not moves:
        return None
    for col in moves:
        if winner(play(board, col, -1)) == -1:
            return col
    for col in moves:
        if winner(play(board, col, 1)) == 1:
            return col
    return next(col for col in _CENTRE_ORDER if col in moves)
