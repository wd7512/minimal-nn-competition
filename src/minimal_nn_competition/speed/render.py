"""Render the hidden self-applied Snake Simple 8x8 speed leaderboard."""

import html
import json
from dataclasses import asdict

from minimal_nn_competition.leaderboard import format_measured, short_commit
from minimal_nn_competition.leaderboard.theme import (
    AUXILIARY_TABLE_CSS,
    SITE_BASE_CSS,
    SITE_THEME_CSS,
)
from minimal_nn_competition.speed.leaderboard import rank_speed_results

_STYLE = (
    SITE_THEME_CSS
    + SITE_BASE_CSS
    + AUXILIARY_TABLE_CSS
    + """
body { margin: 0 auto; max-width: 72rem; padding: 2rem; }
"""
)


def _table(results):
    rows = []
    for rank, result in enumerate(results, 1):
        commit = short_commit(result.commit)
        measured = format_measured(result.measured_at)
        rows.append(
            f"""<tr><td>{rank}</td><td><code title="{html.escape(result.commit)}">{html.escape(commit)}</code></td><td>{result.ticks_per_second:,.0f} ± {result.ticks_std:,.0f}</td><td>{html.escape(result.harness)}</td><td>{html.escape(measured)}</td><td>{html.escape(result.cpu or result.machine or "-")}</td></tr>"""
        )
    return f"""<table><thead><tr><th>#</th><th>Commit</th><th>Ticks/s</th><th>Harness</th><th>Measured</th><th>CPU</th></tr></thead><tbody>{"".join(rows)}</tbody></table>"""


def render_speed_html(results):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Snake Simple 8x8 Environment Speed</title><style>{_STYLE}</style></head><body><header><h1>Snake Simple 8x8 Environment Speed</h1><p>You found the hidden board. A fixed Hamiltonian-cycle agent plays the same Snake Simple 8x8 environment, always filling the board. Each commit is measured with ten one-second runs; the ± value is the population standard deviation. Higher is better.</p><p class="muted">A tick is one SnakeEngine.step() call (game engine only, excluding observation encoding). Older snake-simple-8x8-speed-v1 entries include the 7-feature observation encoding, so compare within a harness version.</p><p><a href="../../">Main leaderboard</a> · <a href="../">Internal tokens</a> · <a href="leaderboard.json">Raw speed JSON</a></p></header>{_table(results) if results else "<p>No speed results have been recorded yet.</p>"}</body></html>
"""


def render_speed_site(results, output_dir):
    ranked = rank_speed_results(results)
    speed_dir = output_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    (speed_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    index = speed_dir / "index.html"
    index.write_text(render_speed_html(ranked))
    return index
