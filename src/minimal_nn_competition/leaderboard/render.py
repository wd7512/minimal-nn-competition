"""Render ranked results as a static GitHub Pages site."""

import html
import json
import shutil
import statistics
from dataclasses import asdict
from itertools import groupby

from minimal_nn_competition import __version__
from minimal_nn_competition.benchmarks.snake.replay import replay_slug
from minimal_nn_competition.leaderboard import rank_results

_SECTION_TITLES = {
    "fashion-mnist": "FASHION-MNIST COMPRESSION",
    "mnist": "MNIST COMPRESSION",
    "snake-simple-8x8": "SNAKE SIMPLE 8X8",
}

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


def _score_text(result):
    maximum = result.details.get("max_length", 64) if result.benchmark == "snake-simple-8x8" else 1
    mean = result.score / maximum * 100
    if result.benchmark == "snake-simple-8x8":
        return f"{mean:.2f}% ± {result.score_std / maximum * 100:.2f}%"
    digits = 4 if not result.passed and mean >= 99.995 else 2
    return f"{mean:.{digits}f}%"


def _game_length_text(result):
    moves = result.details.get("seed_moves")
    return (
        "-"
        if not isinstance(moves, list) or not moves
        else f"{statistics.fmean(moves):.1f} ± {statistics.pstdev(moves):.1f}"
    )


def _table(results, replay_links=None):
    rows = []
    show_game_length = bool(results) and results[0].benchmark == "snake-simple-8x8"
    show_replay = bool(replay_links)
    for rank, result in enumerate(results, 1):
        normalisation = result.details.get("normalisation")
        link = (replay_links or {}).get(result.submission)
        rows.append(
            f"""<tr{' class="passed"' if result.passed else ""}><td>{rank}</td><td>{html.escape(result.submission)}</td><td>{html.escape(result.name)}</td><td>{html.escape(result.author)}</td><td>{_score_text(result)}</td>{f"<td>{_game_length_text(result)}</td>" if show_game_length else ""}<td>{"pass" if result.passed else "fail"}</td><td>{str(normalisation).lower() if isinstance(normalisation, bool) else "-"}</td><td>{result.parameter_count}</td>{f'<td><a href="{html.escape(link)}">view game</a></td>' if link else "<td>-</td>" if show_replay else ""}</tr>"""
        )
    return f"""<table><thead><tr><th>#</th><th>Submission</th><th>Name</th><th>Author</th><th>Score</th>{"<th>Game length</th>" if show_game_length else ""}<th>Status</th><th>Normalisation</th><th>Parameters</th>{"<th>Replay</th>" if show_replay else ""}</tr></thead><tbody>{"".join(rows)}</tbody></table>"""


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


def render_game_html(submission, payload):
    """Render a standalone replay page with an inline frame player."""
    title = html.escape(submission)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Snake Simple 8x8 replay — {title}</title><style>{_GAME_STYLE}</style></head><body><header><h1>Snake Simple 8x8 replay — {title}</h1><p>Seed {int(payload.get("seed", 0))}, {int(payload.get("moves", 0))} moves, final length {int(payload.get("length", 0))}.</p><p><a href="../">Main leaderboard</a></p></header><canvas id="board" width="320" height="320"></canvas><div class="controls"><button id="play" type="button">Play</button><button id="prev" type="button">Step back</button><button id="next" type="button">Step forward</button><label>Speed <select id="speed"><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select></label></div><p id="status" class="muted"></p><script>{_GAME_SCRIPT.replace("__FRAMES__", json.dumps(payload["frames"])).replace("__SIZE__", str(int(payload.get("size", 8))))}</script></body></html>
"""


def render_game_pages(results, replays, output_dir):
    """Write one replay page per snake submission; return submission-to-href links.

    ``replays`` is keyed by the replay slug, not the raw submission name.
    """
    games_dir = output_dir / "games"
    links = {}
    if not replays:
        return links
    shutil.rmtree(games_dir, ignore_errors=True)
    seen = {}
    for result in results:
        if result.benchmark != "snake-simple-8x8":
            continue
        slug = replay_slug(result.submission)
        payload = replays.get(slug)
        if payload is None:
            continue
        previous = seen.get(slug)
        if previous is not None and previous != result.submission:
            raise ValueError(
                f"Replay filename collision: {previous!r} and {result.submission!r} both map to {slug!r}"
            )
        seen[slug] = result.submission
        games_dir.mkdir(parents=True, exist_ok=True)
        (games_dir / f"{slug}.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
        (games_dir / f"{slug}.html").write_text(render_game_html(result.submission, payload))
        links[result.submission] = f"games/{slug}.html"
    return links


def render_html(results, replay_links=None):
    sections = []
    for benchmark, group in groupby(results, key=lambda result: result.benchmark):
        entries = list(group)
        ranking = (
            "pass status, score, parameter count, then game length"
            if benchmark == "snake-simple-8x8"
            else "pass status, score, then parameter count"
        )
        sections.append(
            f"""<section><h2>{html.escape(_SECTION_TITLES.get(benchmark, benchmark.upper()))}</h2><p class="muted">{len(entries)} result(s), ranked by {ranking}.</p>{_table(entries, replay_links if benchmark == "snake-simple-8x8" else None)}</section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Minimal NN Competition</title><style>{_STYLE}</style></head><body><header><h1>Minimal NN Competition</h1><p>Small PyTorch state-dict submissions for MNIST, Fashion-MNIST, and Snake Simple 8x8.</p><p>Submit a model by opening a pull request against the <code>submission</code> branch. See <a href="https://github.com/wd7512/minimal-nn-competition/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>.</p><p><a href="leaderboard.json">Raw leaderboard JSON</a></p></header>{"".join(sections) or "<p>No results have been recorded yet.</p>"}<footer><a class="version" href="internal/speed/" title="Hidden Snake Simple 8x8 environment speed leaderboard" aria-label="Hidden Snake Simple 8x8 environment speed leaderboard">v{html.escape(__version__)}</a></footer></body></html>
"""


def render_site(results, output_dir, replays=None):
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
