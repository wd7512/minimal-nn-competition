"""Render the internal token-count leaderboard as a static site section."""

import html
import json
from dataclasses import asdict
from itertools import groupby

from minimal_nn_competition.internal import rank_internal_results, scope_rank
from minimal_nn_competition.leaderboard import format_measured, short_commit
from minimal_nn_competition.leaderboard.theme import (
    AUXILIARY_TABLE_CSS,
    SITE_BASE_CSS,
    SITE_THEME_CSS,
)

_SCOPE_TITLES = {"snake-simple-8x8": "SNAKE SIMPLE 8X8"}

_STYLE = SITE_THEME_CSS + SITE_BASE_CSS + AUXILIARY_TABLE_CSS + """
body { margin: 0 auto; max-width: 72rem; padding: 2rem; }
.failed { opacity: 0.72; }
"""


def _table(results):
    rows = []
    for result in results:
        commit = short_commit(result.commit)
        measured = format_measured(result.measured_at)
        rows.append(
            f"""<tr><td>{scope_rank(result, results)}</td><td>{result.tokens:,}</td><td><code title="{html.escape(result.commit)}">{html.escape(commit)}</code>{' <span class="muted">(dirty)</span>' if result.dirty else ""}</td><td>{html.escape(measured)}</td><td>{html.escape(result.machine or "-")}</td></tr>"""
            if result.status == "pass"
            else f"""<tr class="failed"><td>-</td><td>-</td><td><code title="{html.escape(result.commit)}">{html.escape(commit)}</code></td><td>{html.escape(measured)}</td><td title="{html.escape(result.failure_reason or "validation failed")}">failed validation</td></tr>"""
        )
    return f"""<table><thead><tr><th>Rank</th><th>Tokens</th><th>Commit</th><th>Measured</th><th>Machine</th></tr></thead><tbody>{"".join(rows)}</tbody></table>"""


def render_internal_html(results):
    ranked = rank_internal_results(results)
    sections = []
    for scope, group in groupby(ranked, key=lambda result: result.scope):
        entries = list(group)
        sections.append(
            f"""<section><h2>{html.escape(_SCOPE_TITLES.get(scope, scope.upper()))}</h2><p class="muted">{sum(1 for result in entries if result.status == "pass")} scored run(s), ranked by ascending token count. Lower is better; equal counts share a rank.</p>{_table(entries)}</section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Internal Token Leaderboard</title><style>{_STYLE}</style></head><body><header><h1>Internal Token Leaderboard</h1><p>Syntactic compactness of this repo's own code: keywords, operators, and literals count; identifiers, comments, whitespace, docstrings, and type annotations do not. A scope scores only when all of its mapped tests pass.</p><p><a href="../">Main leaderboard</a> · <a href="speed/">Speed board</a> · <a href="leaderboard.json">Raw internal JSON</a></p></header>{"".join(sections) or "<p>No internal results have been recorded yet.</p>"}</body></html>
"""


def render_internal_site(results, output_dir):
    ranked = rank_internal_results(results)
    internal_dir = output_dir / "internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    (internal_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    index = internal_dir / "index.html"
    index.write_text(render_internal_html(ranked))
    return index
