"""Render ranked results as a static GitHub Pages site."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from typing import Any

from minimal_nn_competition import __version__
from minimal_nn_competition.benchmarks.snake.replay import replay_slug
from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.runner.entry import BenchmarkResult

_SECTION_TITLES = {"mnist": "MNIST COMPRESSION", "snake": "SNAKE GAME"}

_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 0 auto; max-width: 72rem; padding: 2rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { border-bottom: 1px solid #8884; padding: 0.5rem; text-align: left; }
th { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
tr.passed td { font-weight: 600; }
.muted { opacity: 0.65; }
footer { margin-top: 2rem; text-align: right; }
.version { color: inherit; opacity: 0.35; text-decoration: none; font-size: 0.8rem; }
.version:hover, .version:focus-visible { opacity: 1; }
"""


def _score_text(result: BenchmarkResult) -> str:
    maximum = float(result.details.get("max_length", 64)) if result.benchmark == "snake" else 1.0
    mean = result.score / maximum * 100
    if result.benchmark != "snake":
        return f"{mean:.2f}%"
    std = result.score_std / maximum * 100
    return f"{mean:.2f}% ± {std:.2f}%"


def _table(results: list[BenchmarkResult], replay_links: dict[str, str] | None = None) -> str:
    rows = []
    show_replay = bool(replay_links)
    for rank, result in enumerate(results, start=1):
        status = "pass" if result.passed else "fail"
        normalisation = result.details.get("normalisation")
        normalisation_text = str(normalisation).lower() if isinstance(normalisation, bool) else "-"
        css = ' class="passed"' if result.passed else ""
        row = (
            f"<tr{css}><td>{rank}</td>"
            f"<td>{html.escape(result.submission)}</td>"
            f"<td>{html.escape(result.name)}</td>"
            f"<td>{html.escape(result.author)}</td>"
            f"<td>{_score_text(result)}</td>"
            f"<td>{status}</td>"
            f"<td>{normalisation_text}</td>"
            f"<td>{result.parameter_count}</td>"
        )
        if show_replay:
            link = (replay_links or {}).get(result.submission)
            cell = f'<a href="{html.escape(link)}">view game</a>' if link else "-"
            row += f"<td>{cell}</td>"
        row += "</tr>"
        rows.append(row)
    header = (
        "<table><thead><tr><th>#</th><th>Submission</th><th>Name</th><th>Author</th>"
        "<th>Score</th><th>Status</th><th>Normalisation</th><th>Parameters</th>"
    )
    if show_replay:
        header += "<th>Replay</th>"
    header += "</tr></thead>"
    return header + f"<tbody>{''.join(rows)}</tbody></table>"


_GAME_STYLE = (
    _STYLE
    + """
canvas { border: 1px solid #8884; image-rendering: pixelated; width: min(80vw, 480px);
  height: auto; }
.controls { display: flex; gap: 0.5rem; align-items: center; margin: 1rem 0; flex-wrap: wrap; }
button, select { font: inherit; padding: 0.25rem 0.75rem; }
"""
)

_GAME_SCRIPT = """
const FRAMES = __FRAMES__;
const SIZE = __SIZE__;
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const playButton = document.getElementById("play");
const speed = document.getElementById("speed");
let frame = 0;
let playing = false;
let timer = null;
function draw() {
  const cell = canvas.width / SIZE;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const food = FRAMES[frame].food;
  ctx.fillStyle = "#c33";
  ctx.fillRect(food[0] * cell, food[1] * cell, cell, cell);
  const snake = FRAMES[frame].snake;
  snake.forEach((segment, index) => {
    ctx.fillStyle = index === 0 ? "#1a7f37" : "#3fb950";
    ctx.fillRect(segment[0] * cell, segment[1] * cell, cell, cell);
  });
  status.textContent = "move " + frame + "/" + (FRAMES.length - 1)
    + ", length " + snake.length;
}
function step(delta) {
  frame = Math.min(FRAMES.length - 1, Math.max(0, frame + delta));
  draw();
}
function stop() {
  playing = false;
  playButton.textContent = "Play";
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}
function play() {
  if (playing) {
    stop();
    return;
  }
  if (frame >= FRAMES.length - 1) {
    frame = 0;
  }
  playing = true;
  playButton.textContent = "Pause";
  timer = setInterval(() => {
    frame += 1;
    if (frame >= FRAMES.length - 1) {
      frame = FRAMES.length - 1;
      stop();
    }
    draw();
  }, Math.max(16, Math.floor(240 / Number(speed.value))));
}
playButton.addEventListener("click", play);
document.getElementById("prev").addEventListener("click", () => { stop(); step(-1); });
document.getElementById("next").addEventListener("click", () => { stop(); step(1); });
speed.addEventListener("change", () => { if (playing) { stop(); play(); } });
draw();
"""


def render_game_html(submission: str, payload: dict[str, Any]) -> str:
    """Render a standalone replay page with an inline frame player."""
    size = int(payload.get("size", 8))
    frames_json = json.dumps(payload["frames"])
    script = _GAME_SCRIPT.replace("__FRAMES__", frames_json).replace("__SIZE__", str(size))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Snake replay — {html.escape(submission)}</title>"
        f"<style>{_GAME_STYLE}</style></head><body>"
        f"<header><h1>Snake replay — {html.escape(submission)}</h1>"
        f"<p>Seed {int(payload.get('seed', 0))}, {int(payload.get('moves', 0))} moves, "
        f"final length {int(payload.get('length', 0))}.</p>"
        '<p><a href="../">Main leaderboard</a></p></header>'
        '<canvas id="board" width="320" height="320"></canvas>'
        '<div class="controls"><button id="play" type="button">Play</button>'
        '<button id="prev" type="button">Step back</button>'
        '<button id="next" type="button">Step forward</button>'
        '<label>Speed <select id="speed">'
        '<option value="1">1x</option><option value="2">2x</option>'
        '<option value="4" selected>4x</option><option value="8">8x</option>'
        "</select></label></div>"
        '<p id="status" class="muted"></p>'
        f"<script>{script}</script>"
        "</body></html>\n"
    )


def render_game_pages(
    results: list[BenchmarkResult],
    replays: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    """Write one replay page per snake submission; return submission-to-href links.

    ``replays`` is keyed by the replay slug, not the raw submission name.
    """
    games_dir = output_dir / "games"
    links: dict[str, str] = {}
    if not replays:
        return links
    shutil.rmtree(games_dir, ignore_errors=True)
    seen: dict[str, str] = {}
    for result in results:
        if result.benchmark != "snake":
            continue
        slug = replay_slug(result.submission)
        payload = replays.get(slug)
        if payload is None:
            continue
        previous = seen.get(slug)
        if previous is not None and previous != result.submission:
            raise ValueError(
                f"Replay filename collision: {previous!r} and {result.submission!r} "
                f"both map to {slug!r}"
            )
        seen[slug] = result.submission
        games_dir.mkdir(parents=True, exist_ok=True)
        (games_dir / f"{slug}.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
        (games_dir / f"{slug}.html").write_text(render_game_html(result.submission, payload))
        links[result.submission] = f"games/{slug}.html"
    return links


def render_html(results: list[BenchmarkResult], replay_links: dict[str, str] | None = None) -> str:
    sections = []
    for benchmark, group in groupby(results, key=lambda result: result.benchmark):
        entries = list(group)
        title = _SECTION_TITLES.get(benchmark, benchmark.upper())
        links = replay_links if benchmark == "snake" else None
        sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f'<p class="muted">{len(entries)} result(s), ranked by pass status, '
            "score, then parameter count.</p>"
            f"{_table(entries, links)}</section>"
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
        "<p>Submit a model by opening a pull request against the "
        '<code>submission</code> branch. See <a href="'
        "https://github.com/wd7512/minimal-nn-competition/blob/main/CONTRIBUTING.md"
        '">CONTRIBUTING.md</a>.</p>'
        '<p><a href="leaderboard.json">Raw leaderboard JSON</a></p></header>'
        f"{''.join(sections)}"
        "<footer>"
        '<a class="version" href="internal/speed/" title="Hidden Snake environment speed leaderboard" '
        'aria-label="Hidden Snake environment speed leaderboard">'
        f"v{html.escape(__version__)}</a></footer>"
        "</body></html>\n"
    )


def render_site(
    results: list[BenchmarkResult],
    output_dir: Path,
    replays: dict[str, dict[str, Any]] | None = None,
) -> Path:
    ranked = rank_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    replay_links = render_game_pages(ranked, replays or {}, output_dir)
    index = output_dir / "index.html"
    index.write_text(render_html(ranked, replay_links or None))
    (output_dir / ".nojekyll").write_text("")
    return index
