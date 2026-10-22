"""Deterministic token-count scoring for Python solutions (issue #6).

Rules (versioned by ``RULES_VERSION``): every counted token costs 1.
Count operators, punctuation, numeric/string literals, and keywords.
Never count identifiers, comments, whitespace, docstrings, or tokenizer
bookkeeping tokens. Soft keywords (``match``/``case``/``_``/``type``) count
only when used as syntax, anchored via the ``ast`` positions below.
"""

import ast
import io
import keyword
import tokenize

RULES_VERSION = 1


def _docstring_spans(tree):
    """Return the start and end positions of every module/class/function docstring."""
    spans = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        spans.add(
            (
                value.lineno,
                value.col_offset,
                value.end_lineno or value.lineno,
                value.end_col_offset or value.col_offset,
            )
        )
    return spans


def _syntactic_soft_positions(tree, tokens):
    """Return positions where soft keywords act as language syntax.

    ``match`` sits at the start of its ``ast.Match`` node; ``case`` precedes
    its pattern on the same line; ``_`` counts only inside match patterns;
    ``type`` sits at the start of a ``TypeAlias`` node.
    """
    positions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            positions.add((node.lineno, node.col_offset))
        elif isinstance(node, ast.match_case):
            pattern = node.pattern
            line, col = pattern.lineno, pattern.col_offset
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
            end = (pattern.end_lineno or line, pattern.end_col_offset or col)
            for token in tokens:
                if (
                    token.type == tokenize.NAME
                    and token.string == "_"
                    and (line, col) <= token.start < end
                ):
                    positions.add(token.start)
        elif isinstance(node, ast.TypeAlias):
            positions.add((node.lineno, node.col_offset))
    return positions


def count_tokens(source):
    """Count syntax tokens in ``source`` following the module rules."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot score unparseable Python source: {exc}") from exc
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"Cannot tokenize Python source: {exc}") from exc
    docstrings = _docstring_spans(tree)
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
        elif token.type == tokenize.NAME:
            if (
                token.string in keyword.kwlist
                or token.string in keyword.softkwlist
                and token.start in syntactic
            ):
                count += 1
        elif token.type == tokenize.STRING:
            if not any(
                (row, col) <= token.start < (last_row, last_col)
                for row, col, last_row, last_col in docstrings
            ):
                count += 1
        elif token.type in (tokenize.OP, tokenize.NUMBER):
            count += 1
    return count


def score_file(path):
    """Return the token count of one Python file."""
    try:
        return count_tokens(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot score {path}: {exc}") from exc


def score_files(paths):
    """Return the total token count of ``paths``."""
    return sum(score_file(path) for path in sorted(paths))
