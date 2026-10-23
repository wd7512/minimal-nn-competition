"""Pure connect-4 rules and a 1-ply win-block-centre opponent policy."""

ROWS = 6
COLS = 7

_CENTRE_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _lines():
    # (step, rows, cols, first row): across, down, diagonal-down, diagonal-up.
    lines = []
    for step, nrows, ncols, first in (
        (1, 6, 4, 0),
        (7, 3, 7, 0),
        (8, 3, 4, 0),
        (-6, 3, 4, 3),
    ):
        for row in range(first, first + nrows):
            for col in range(ncols):
                base = row * COLS + col
                lines.append((base, base + step, base + 2 * step, base + 3 * step))
    return tuple(lines)


LINES = _lines()


def winner(board):
    """Return 1 (net wins), -1 (opponent wins), 0 (full-board draw), or None."""
    for a, b, c, d in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c] == board[d]:
            return board[a]
    return 0 if 0 not in board else None


def legal_moves(board):
    """Return the non-full columns in ascending order."""
    return [col for col in range(COLS) if not board[col]]


def play(board, col, player):
    """Drop ``player``'s disc into ``col``; the disc lands on the lowest empty row."""
    cells = list(board)
    for row in reversed(range(ROWS)):
        index = row * COLS + col
        if not cells[index]:
            cells[index] = player
            return tuple(cells)
    raise ValueError(f"Column {col} is full")


def opponent_move(board):
    """Win if possible, else block the net's immediate win, else centre-first pick."""
    moves = legal_moves(board)
    if not moves:
        return None
    # Try to win first, then block the net's immediate win.
    for player in (-1, 1):
        for col in moves:
            if winner(play(board, col, player)) == player:
                return col
    return next(col for col in _CENTRE_ORDER if col in moves)
