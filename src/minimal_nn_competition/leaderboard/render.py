"""Render ranked results as a static GitHub Pages site."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from itertools import groupby
from pathlib import Path

from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.runner.entry import BenchmarkResult

_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 0 auto; max-width: 72rem; padding: 2rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { border-bottom: 1px solid #8884; padding: 0.5rem; text-align: left; }
th { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
tr.passed td { font-weight: 600; }
pre { margin: 0; white-space: pre-wrap; font-size: 0.8rem; }
details summary { cursor: pointer; }
.muted { opacity: 0.65; }
"""


def _table(results: list[BenchmarkResult]) -> str:
    rows = []
    for rank, result in enumerate(results, start=1):
        details = html.escape(json.dumps(result.details, sort_keys=True))
        status = "pass" if result.passed else "fail"
        normalisation = result.details.get("normalisation")
        normalisation_text = str(normalisation).lower() if isinstance(normalisation, bool) else "-"
        css = ' class="passed"' if result.passed else ""
        rows.append(
            f"<tr{css}><td>{rank}</td>"
            f"<td>{html.escape(result.submission)}</td>"
            f"<td>{result.score:.4f}</td>"
            f"<td>{status}</td>"
            f"<td>{normalisation_text}</td>"
            f"<td>{result.parameter_count}</td>"
            f"<td><details><summary>details</summary><pre>{details}</pre></details></td></tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>Submission</th><th>Score</th>"
        "<th>Status</th><th>Normalisation</th><th>Parameters</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(results: list[BenchmarkResult]) -> str:
    sections = []
    for benchmark, group in groupby(results, key=lambda result: result.benchmark):
        entries = list(group)
        sections.append(
            f"<section><h2>{html.escape(benchmark.upper())}</h2>"
            f'<p class="muted">{len(entries)} result(s), ranked by pass status, '
            "score, then parameter count.</p>"
            f"{_table(entries)}</section>"
        )
    if not sections:
        sections.append("<p>No results have been recorded yet.</p>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Minimal NN Competition</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1>Minimal NN Competition</h1>"
        "<p>Small PyTorch state-dict submissions for MNIST and Snake.</p>"
        '<p><a href="leaderboard.json">Raw leaderboard JSON</a></p></header>'
        f"{''.join(sections)}"
        "</body></html>\n"
    )


def render_site(results: list[BenchmarkResult], output_dir: Path) -> Path:
    ranked = rank_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    index = output_dir / "index.html"
    index.write_text(render_html(ranked))
    (output_dir / ".nojekyll").write_text("")
    return index
