"""Deterministic token-count scoring for Python solutions (issue #6).

Rules (versioned by ``RULES_VERSION``): every counted token costs 1.
Count operators, punctuation, numeric/string literals, and keywords.
Never count identifiers, comments, whitespace, docstrings, or tokenizer
bookkeeping tokens. Soft keywords (``match``/``case``/``_``/``type``) count
only when used as syntax, anchored via ``ast`` positions.
"""

from __future__ import annotations

import ast
import io
import keyword
import tokenize
from pathlib import Path

RULES_VERSION = 1

_SKIPPED_TYPES = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.FSTRING_MIDDLE,
        tokenize.FSTRING_END,
    }
)

# Soft keywords count only at positions anchored to real syntax (see
# _syntactic_soft_positions); ordinary uses of these names as identifiers
# never count.
_SOFT_KEYWORDS = frozenset(keyword.softkwlist)


def _docstring_starts(tree: ast.AST) -> set[tuple[int, int]]:
    """Return (row, col) of every module/class/function docstring."""
    starts: set[tuple[int, int]] = set()

    def _visit(node: ast.AST) -> None:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                starts.add((first.value.lineno, first.value.col_offset))
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return starts


def _syntactic_soft_positions(
    tree: ast.AST, tokens: list[tokenize.TokenInfo]
) -> set[tuple[int, int]]:
    """Return positions where soft keywords act as language syntax.

    ``match`` sits at the start of its ``ast.Match`` node; ``case`` precedes
    its pattern on the same line; ``_`` counts inside a match statement span;
    ``type`` sits at the start of a ``TypeAlias`` node.
    """
    positions: set[tuple[int, int]] = set()
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Match)]
    for node in matches:
        positions.add((node.lineno, node.col_offset))
    cases = [node for node in ast.walk(tree) if isinstance(node, ast.match_case)]
    for case in cases:
        line, col = case.pattern.lineno, case.pattern.col_offset
        starts = [
            token.start
            for token in tokens
            if token.type == tokenize.NAME
            and token.string == "case"
            and token.start[0] == line
            and token.start[1] < col
        ]
        if starts:
            positions.add(max(starts))
    spans = [(node.lineno, node.end_lineno or node.lineno) for node in matches]
    for token in tokens:
        if token.type != tokenize.NAME or token.string != "_":
            continue
        if any(first <= token.start[0] <= last for first, last in spans):
            positions.add(token.start)
    for node in ast.walk(tree):
        if isinstance(node, ast.TypeAlias):
            positions.add((node.lineno, node.col_offset))
    return positions


def count_tokens(source: str) -> int:
    """Count syntax tokens in ``source`` following the module rules."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot score unparseable Python source: {exc}") from exc
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"Cannot tokenize Python source: {exc}") from exc
    docstrings = _docstring_starts(tree)
    syntactic = _syntactic_soft_positions(tree, tokens)
    count = 0
    fstring_depth = 0
    for token in tokens:
        if fstring_depth:
            if token.type == tokenize.FSTRING_START:
                fstring_depth += 1
            elif token.type == tokenize.FSTRING_END:
                fstring_depth -= 1
            continue
        if token.type == tokenize.FSTRING_START:
            # A whole f-string (including any embedded expressions) is one token.
            count += 1
            fstring_depth = 1
        elif token.type in _SKIPPED_TYPES:
            continue
        elif token.type == tokenize.NAME:
            if token.string in keyword.kwlist or token.string in _SOFT_KEYWORDS and token.start in syntactic:
                count += 1
        elif token.type == tokenize.STRING:
            if token.start not in docstrings:
                count += 1
        elif token.type in (tokenize.OP, tokenize.NUMBER):
            count += 1
    return count


def score_file(path: Path) -> int:
    """Return the token count of one Python file."""
    try:
        return count_tokens(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot score {path}: {exc}") from exc


def score_files(paths: list[Path]) -> int:
    """Return the total token count of ``paths``."""
    return sum(score_file(path) for path in sorted(paths))
