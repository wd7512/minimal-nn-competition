"""Render the hidden self-applied Snake speed leaderboard."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from minimal_nn_competition.speed.benchmark import SpeedResult
from minimal_nn_competition.speed.leaderboard import rank_speed_results

_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 0 auto; max-width: 72rem; padding: 2rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { border-bottom: 1px solid #8884; padding: 0.5rem; text-align: left; }
th { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
.muted { opacity: 0.65; }
"""


def _commit_text(commit: str) -> str:
    return commit[:7] if commit != "unknown" else commit


def _measured_text(measured_at: str) -> str:
    return measured_at.replace("T", " ").replace("+00:00", " UTC") or "-"


def _table(results: list[SpeedResult]) -> str:
    rows = []
    for rank, result in enumerate(results, start=1):
        commit = html.escape(_commit_text(result.commit))
        measured = html.escape(_measured_text(result.measured_at))
        rows.append(
            f"<tr><td>{rank}</td>"
            f'<td><code title="{html.escape(result.commit)}">{commit}</code></td>'
            f"<td>{result.ticks_per_second:,.0f} ± {result.ticks_std:,.0f}</td>"
            f"<td>{measured}</td>"
            f"<td>{html.escape(result.cpu or result.machine or '-')}</td></tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>Commit</th><th>Ticks/s</th>"
        "<th>Measured</th><th>CPU</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_speed_html(results: list[SpeedResult]) -> str:
    content = _table(results) if results else "<p>No speed results have been recorded yet.</p>"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="robots" content="noindex,nofollow">'
        "<title>Snake Environment Speed</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1>Snake Environment Speed</h1>"
        "<p>You found the hidden board. A fixed Hamiltonian-cycle agent plays the same "
        "8x8 Snake environment, always filling the board. Each commit is measured with "
        "ten one-second runs; the ± value is the population standard deviation. Higher "
        "is better.</p>"
        '<p class="muted">A tick is one SnakeEnvironment.step() call.</p>'
        '<p><a href="../">Main leaderboard</a> · '
        '<a href="leaderboard.json">Raw speed JSON</a></p></header>'
        f"{content}"
        "</body></html>\n"
    )


def render_speed_site(results: list[SpeedResult], output_dir: Path) -> Path:
    ranked = rank_speed_results(results)
    speed_dir = output_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    (speed_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    index = speed_dir / "index.html"
    index.write_text(render_speed_html(ranked))
    return index
