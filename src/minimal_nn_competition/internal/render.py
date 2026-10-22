"""Render the internal token-count leaderboard as a static site section."""

import html
import json
from dataclasses import asdict
from itertools import groupby

from minimal_nn_competition.internal import rank_internal_results, scope_rank

_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 0 auto; max-width: 72rem; padding: 2rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { border-bottom: 1px solid #8884; padding: 0.5rem; text-align: left; }
th { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
.muted { opacity: 0.65; }
.failed { opacity: 0.65; }
"""


def _commit_text(commit):
    return commit[:7] if commit != "unknown" else commit


def _measured_text(measured_at):
    return measured_at.replace("T", " ").replace("+00:00", " UTC") or "-"


def _table(results):
    rows = []
    for result in results:
        commit = html.escape(_commit_text(result.commit))
        measured = html.escape(_measured_text(result.measured_at))
        if result.status == "pass":
            rows.append(
                f"<tr><td>{scope_rank(result, results)}</td>"
                f"<td>{result.tokens:,}</td>"
                f'<td><code title="{html.escape(result.commit)}">{commit}</code>'
                f"{' <span class="muted">(dirty)</span>' if result.dirty else ''}</td>"
                f"<td>{measured}</td>"
                f"<td>{html.escape(result.machine or '-')}</td></tr>"
            )
        else:
            reason = html.escape(result.failure_reason or "validation failed")
            rows.append(
                f'<tr class="failed"><td>-</td><td>-</td>'
                f'<td><code title="{html.escape(result.commit)}">{commit}</code></td>'
                f"<td>{measured}</td>"
                f'<td title="{reason}">failed validation</td></tr>'
            )
    return (
        "<table><thead><tr><th>Rank</th><th>Tokens</th><th>Commit</th>"
        "<th>Measured</th><th>Machine</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_internal_html(results):
    ranked = rank_internal_results(results)
    sections = []
    for scope, group in groupby(ranked, key=lambda result: result.scope):
        entries = list(group)
        passes = sum(1 for result in entries if result.status == "pass")
        sections.append(
            f"<section><h2>{html.escape(scope.upper())}</h2>"
            f'<p class="muted">{passes} scored run(s), ranked by ascending token '
            "count. Lower is better; equal counts share a rank.</p>"
            f"{_table(entries)}</section>"
        )
    if not sections:
        sections.append("<p>No internal results have been recorded yet.</p>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="robots" content="noindex,nofollow">'
        "<title>Internal Token Leaderboard</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1>Internal Token Leaderboard</h1>"
        "<p>Syntactic compactness of this repo's own code: keywords, operators, "
        "and literals count; identifiers, comments, whitespace, and docstrings "
        "do not. A scope scores only when all of its mapped tests pass.</p>"
        '<p><a href="../">Main leaderboard</a> · '
        '<a href="speed/">Speed board</a> · '
        '<a href="leaderboard.json">Raw internal JSON</a></p></header>'
        f"{''.join(sections)}"
        "</body></html>\n"
    )


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
