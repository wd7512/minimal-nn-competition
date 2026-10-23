"""Render ranked results as a static GitHub Pages site."""

import html
import json
import shutil
import statistics
from dataclasses import asdict

from minimal_nn_competition import __version__
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.benchmarks.snake.replay import replay_stem
from minimal_nn_competition.leaderboard import rank_results

_SECTION_TITLES = {
    "chess-eval": "CHESS EVALUATION",
    "fashion-mnist": "FASHION-MNIST COMPRESSION",
    "mnist": "MNIST COMPRESSION",
    "snake-simple-8x8": "SNAKE SIMPLE 8X8",
    "snake-simple-16x16": "SNAKE SIMPLE 16X16",
    "snake-obstacles-8x8": "SNAKE OBSTACLES 8X8",
}
_SNAKE_MAX_LENGTH = {
    "snake-simple-8x8": 64,
    "snake-simple-16x16": 256,
    "snake-obstacles-8x8": 64,
}

_GROUP_SPECS = (
    {
        "id": "snake",
        "label": "Snake",
        "eyebrow": "Games",
        "description": (
            "Three board variants: open 8×8, open 16×16, and 8×8 with deterministic blocked cells."
        ),
        "benchmarks": (
            "snake-simple-8x8",
            "snake-simple-16x16",
            "snake-obstacles-8x8",
        ),
        "accent": "snake",
    },
    {
        "id": "mnist",
        "label": "MNIST",
        "eyebrow": "Vision",
        "description": "Two 70,000-example image classification tasks.",
        "benchmarks": ("mnist", "fashion-mnist"),
        "accent": "mnist",
    },
    {
        "id": "chess",
        "label": "Chess evaluation",
        "eyebrow": "Regression",
        "description": "Predict Stockfish evaluation within 0.1 pawns on 10,000 positions.",
        "benchmarks": ("chess-eval",),
        "accent": "chess",
    },
)
_VERSION_LABELS = {
    "snake-simple-8x8": "Simple 8×8",
    "snake-simple-16x16": "Simple 16×16",
    "snake-obstacles-8x8": "Obstacles 8×8",
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "chess-eval": "Stockfish 0.1 pawn",
}
_PANEL_NOTES = {
    "snake-simple-8x8": "64-cell open board",
    "snake-simple-16x16": "256-cell open board",
    "snake-obstacles-8x8": "Blocked cells, closed-tour validation",
    "mnist": "Handwritten digits",
    "fashion-mnist": "Garment categories",
    "chess-eval": "10,000 fixed positions",
}

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f4f6f4;
  --surface: #fbfcfa;
  --ink: #17231f;
  --muted: #63706b;
  --rule: #d9e0db;
  --rule-strong: #aebbb5;
  --hover: #edf2ef;
  --pass: #1f6b43;
  --fail: #8a3434;
  --focus: #2f5f57;
  --accent-snake: #256b66;
  --accent-mnist: #8a5b16;
  --accent-chess: #3f5f86;
  --display: ui-serif, Georgia, "Times New Roman", serif;
  --body: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1513;
    --surface: #131d1a;
    --ink: #e7efeb;
    --muted: #9aaba4;
    --rule: #273631;
    --rule-strong: #40534c;
    --hover: #182622;
    --pass: #74cf9b;
    --fail: #f09b91;
    --focus: #8bcacb;
    --accent-snake: #7fcac1;
    --accent-mnist: #e0b76b;
    --accent-chess: #9bb9df;
  }
}
*, *::before, *::after { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--body);
  font-size: 16px;
  line-height: 1.55;
}
a { color: inherit; text-underline-offset: 3px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 3px; }
[data-accent="snake"] { --accent: var(--accent-snake); }
[data-accent="mnist"] { --accent: var(--accent-mnist); }
[data-accent="chess"] { --accent: var(--accent-chess); }
[data-accent="default"] { --accent: var(--accent-snake); }
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font: 700 9px/1 var(--mono);
  letter-spacing: .08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.status::before { content: ""; width: 5px; height: 5px; background: currentColor; flex: none; }
.status--pass { color: var(--pass); }
.status--fail { color: var(--fail); }
"""

_SITE_STYLE = """
.shell {
  width: 100%;
  max-width: 1280px;
  margin: 0 auto;
  padding: 48px 32px 80px;
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 64px;
}
.rail { position: sticky; top: 32px; align-self: start; }
.brand-mark {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font: 700 11px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--ink);
}
.brand-dots { display: inline-flex; gap: 3px; }
.brand-dots span { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
.brand-dots span:nth-child(1) { background: var(--accent-snake); }
.brand-dots span:nth-child(2) { background: var(--accent-mnist); }
.brand-dots span:nth-child(3) { background: var(--accent-chess); }
.rail__label {
  margin: 24px 0 8px;
  font: 700 10px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
.rail nav { display: grid; border-top: 1px solid var(--rule); }
.rail a {
  display: grid;
  grid-template-columns: 6px 22px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--rule);
  color: var(--ink);
  text-decoration: none;
}
.rail__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.rail__index { font: 10px/1.4 var(--mono); color: var(--muted); }
.rail a strong { font-weight: 600; }
.rail a:hover strong { text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: 3px; }
.rail a small { font: 10px/1.4 var(--mono); color: var(--muted); white-space: nowrap; }
.rail__note { margin: 20px 0 0; font: 11px/1.5 var(--mono); color: var(--muted); }
.main { min-width: 0; }
.masthead { border-bottom: 1px solid var(--rule-strong); padding-bottom: 32px; }
.eyebrow {
  margin: 0;
  font: 700 10px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
h1 {
  margin: 16px 0 0;
  font: 400 clamp(2.8rem, 5vw, 4.8rem)/.96 var(--display);
  letter-spacing: -.04em;
  max-width: 13ch;
}
.lead { margin: 20px 0 0; max-width: 58ch; color: var(--muted); font-size: 18px; }
.links {
  margin: 16px 0 0;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  font: 11px/1.4 var(--mono);
}
.links span { color: var(--muted); }
.group { padding: 48px 0 24px; }
.group + .group { border-top: 1px solid var(--rule); padding-top: 48px; }
.group:last-child { padding-bottom: 48px; }
.group__head {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 16px 32px;
  align-items: start;
}
.group__index { color: var(--accent); }
.group__title {
  margin: 8px 0 0;
  font: 400 clamp(2rem, 3.6vw, 3.2rem)/1 var(--display);
  letter-spacing: -.03em;
}
.group__title-trigger {
  display: inline-flex;
  align-items: baseline;
  gap: 12px;
}
.js .group__title-trigger.is-clickable { cursor: pointer; }
.js .group__title-trigger.is-clickable:hover .group__title-text {
  text-decoration: underline;
  text-decoration-color: var(--accent);
  text-underline-offset: 5px;
}
.cycle { display: none; font: 12px/1 var(--mono); color: var(--accent); }
.js .cycle { display: inline; }
.group__desc { margin: 12px 0 0; max-width: 58ch; color: var(--muted); }
.group__stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  font: 11px/1.4 var(--mono);
  color: var(--muted);
  white-space: nowrap;
}
.group__stats strong {
  color: var(--ink);
  font-weight: 600;
  font-size: 15px;
  font-variant-numeric: tabular-nums;
}
.version-row {
  display: none;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px 16px;
  margin-top: 20px;
  padding-top: 12px;
  border-top: 1px solid var(--rule);
}
.js .version-row { display: flex; }
.version-row__label {
  margin: 0;
  font: 700 10px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
.version-links { display: flex; flex-wrap: wrap; gap: 4px 18px; align-items: baseline; }
.version-link {
  appearance: none;
  border: 0;
  border-bottom: 1px solid transparent;
  background: none;
  color: var(--muted);
  padding: 0 0 3px;
  font: 600 13px/1.4 var(--mono);
  cursor: pointer;
}
.version-link:hover { color: var(--ink); }
.version-link.is-active { color: var(--ink); border-bottom-color: var(--accent); }
.version-hint { margin: 0; font: 10px/1.4 var(--mono); color: var(--muted); }
.version-panel { margin-top: 20px; }
.version-panel + .version-panel { margin-top: 40px; }
.js .version-panel:not(.is-active) { display: none; }
.table-note { margin: 0 0 8px; font: 10px/1.4 var(--mono); color: var(--muted); }
.table-note strong { color: var(--ink); font-weight: 600; }
.table-wrap {
  overflow-x: auto;
  border-top: 1px solid var(--rule-strong);
  border-bottom: 1px solid var(--rule-strong);
  -webkit-overflow-scrolling: touch;
}
table.results { width: 100%; border-collapse: collapse; min-width: 40rem; }
.results th, .results td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--rule);
  text-align: left;
  vertical-align: top;
}
.results thead th {
  background: var(--surface);
  font: 700 9px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
}
.results tbody tr:last-child td { border-bottom: 0; }
.results tbody tr:hover td { background: var(--hover); }
.results th:first-child, .results td:first-child {
  position: sticky;
  left: 0;
  background: var(--bg);
  z-index: 1;
}
.results thead th:first-child { background: var(--surface); z-index: 3; }
.results tbody tr:hover td:first-child { background: var(--hover); }
td.rank { font: 11px/1.4 var(--mono); color: var(--muted); }
td.submission a {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  color: var(--ink);
  text-decoration: underline;
  text-decoration-color: var(--rule-strong);
  text-underline-offset: 3px;
}
td.submission a:hover { text-decoration-color: var(--accent); text-decoration-thickness: 2px; }
.submission__name { font-weight: 600; }
.submission__id { font: 10px/1.35 var(--mono); color: var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.num { font-family: var(--mono); font-size: 12px; }
td.score { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--ink); }
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
.footer {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: baseline;
  padding-top: 16px;
  border-top: 1px solid var(--rule-strong);
  color: var(--muted);
  font: 10px/1.5 var(--mono);
}
.footer p { margin: 0; }
.version { color: inherit; opacity: .55; text-decoration: none; }
.version:hover, .version:focus-visible { opacity: 1; }
.js-only { display: none; }
.js .js-only { display: inline; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
@media (max-width: 1024px) {
  .shell { grid-template-columns: 1fr; gap: 32px; padding: 32px 24px 64px; }
  .rail { position: static; }
  .rail nav { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .rail a { grid-template-columns: 6px 20px minmax(0, 1fr); border-bottom: 0; padding: 10px 12px 10px 0; }
  .rail a + a { border-left: 1px solid var(--rule); padding-left: 12px; }
  .rail a small { grid-column: 3; }
  .rail__note { display: none; }
}
@media (max-width: 700px) {
  .shell { padding: 24px 16px 56px; }
  .rail nav { grid-template-columns: 1fr; }
  .rail a { border-left: 0; border-bottom: 1px solid var(--rule); padding-left: 0; }
  .rail a + a { border-left: 0; padding-left: 0; }
  .group__head { grid-template-columns: 1fr; }
  .group__stats { flex-direction: row; align-items: baseline; gap: 16px; }
  .footer { flex-direction: column; gap: 8px; }
}
"""

_DETAIL_STYLE = """
.detail-shell { width: 100%; max-width: 1120px; margin: 0 auto; padding: 48px 32px 80px; }
.detail-head { border-bottom: 1px solid var(--rule-strong); padding-bottom: 24px; }
.eyebrow {
  margin: 0;
  font: 700 10px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
.detail-group { color: var(--accent); }
.detail-title {
  margin: 14px 0 0;
  font: 400 clamp(2.2rem, 4.4vw, 3.8rem)/1 var(--display);
  letter-spacing: -.03em;
  max-width: 18ch;
}
.detail-subtitle { margin: 12px 0 0; color: var(--muted); font-size: 16px; }
.detail-links { margin: 16px 0 0; display: flex; flex-wrap: wrap; gap: 8px 16px; font: 11px/1.4 var(--mono); }
.detail-links span { color: var(--muted); }
.detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(260px, .85fr);
  gap: 48px;
  align-items: start;
  margin-top: 32px;
}
.detail-grid--single { grid-template-columns: minmax(0, 42rem); }
.detail-section { min-width: 0; }
.detail-section-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--rule-strong);
}
.detail-section-head h2 { margin: 0; font: 600 16px/1.3 var(--body); }
.detail-section-head p { margin: 0; font: 10px/1.5 var(--mono); color: var(--muted); }
.detail-game canvas {
  display: block;
  margin: 16px 0 0;
  border: 1px solid var(--rule-strong);
  image-rendering: pixelated;
  width: min(100%, 480px);
  height: auto;
}
.controls { display: flex; gap: .5rem; align-items: center; margin: 16px 0 0; flex-wrap: wrap; }
button, select { font: inherit; padding: .25rem .75rem; }
.muted { color: var(--muted); }
.detail-panel { border-top: 1px solid var(--rule-strong); padding-top: 12px; }
.detail-kicker {
  margin: 0 0 8px;
  font: 700 10px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
.detail-meta { margin: 0; }
.detail-meta div {
  display: grid;
  grid-template-columns: minmax(7rem, 10rem) minmax(0, 1fr);
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--rule);
}
.detail-meta div:last-child { border-bottom: 0; }
.detail-meta dt {
  font: 700 9px/1.5 var(--mono);
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
}
.detail-meta dd { margin: 0; font-size: 14px; line-height: 1.5; overflow-wrap: anywhere; }
.detail-score { font-family: var(--mono); font-weight: 600; }
.detail-footer {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: baseline;
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--rule-strong);
  color: var(--muted);
  font: 10px/1.5 var(--mono);
}
.detail-footer a { color: inherit; }
@media (max-width: 800px) {
  .detail-shell { padding: 32px 20px 64px; }
  .detail-grid { grid-template-columns: 1fr; gap: 32px; }
  .detail-grid--single { grid-template-columns: 1fr; }
  .detail-meta div { grid-template-columns: 1fr; gap: 2px; }
  .detail-footer { flex-direction: column; gap: 8px; }
}
"""

_GAME_SCRIPT = """
const FRAMES = __FRAMES__;
const SIZE = __SIZE__;
const BLOCKED = __BLOCKED__;
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
  BLOCKED.forEach((blockedCell) => {
    ctx.fillStyle = "#555";
    ctx.fillRect(blockedCell[0] * cell, blockedCell[1] * cell, cell, cell);
  });
  const food = FRAMES[frame].food;
  if (food) {
    ctx.fillStyle = "#c33";
    ctx.fillRect(food[0] * cell, food[1] * cell, cell, cell);
  }
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

_SITE_SCRIPT = """
(function () {
  const groups = Array.from(document.querySelectorAll("[data-group-section]"));
  groups.forEach(function (group) {
    const panels = Array.from(group.querySelectorAll(".version-panel"));
    if (!panels.length) return;
    const versions = panels.map(function (panel) { return panel.dataset.version; });
    const links = Array.from(group.querySelectorAll("[data-version-button]"));
    const trigger = group.querySelector("[data-cycle-trigger]");
    const current = group.querySelector("[data-current]");
    const count = group.querySelector("[data-count]");
    const countWord = group.querySelector("[data-count-word]");
    const pass = group.querySelector("[data-pass]");
    function select(id) {
      const panel = panels.find(function (item) { return item.dataset.version === id; });
      if (!panel) return;
      const index = versions.indexOf(id) + 1;
      panels.forEach(function (item) {
        item.hidden = item !== panel;
        item.classList.toggle("is-active", item === panel);
      });
      links.forEach(function (link) {
        const on = link.dataset.version === id;
        link.classList.toggle("is-active", on);
        link.setAttribute("aria-pressed", String(on));
      });
      if (current) {
        current.textContent = (panel.dataset.label || "") + " · " + index + " of " + versions.length;
      }
      if (count) count.textContent = panel.dataset.count || "";
      if (countWord) countWord.textContent = Number(panel.dataset.count) === 1 ? "run" : "runs";
      if (pass) pass.textContent = panel.dataset.pass || "";
      if (trigger) {
        trigger.setAttribute(
          "aria-label",
          "Switch " + group.dataset.title + " version. Current: " +
            (panel.dataset.label || "") + ". " + index + " of " + versions.length + "."
        );
      }
    }
    links.forEach(function (link) {
      link.addEventListener("click", function () { select(link.dataset.version); });
    });
    if (trigger && versions.length > 1) {
      trigger.classList.add("is-clickable");
      trigger.setAttribute("role", "button");
      trigger.setAttribute("tabindex", "0");
      const next = function () {
        const active = panels.find(function (panel) { return !panel.hidden; }) || panels[0];
        const index = versions.indexOf(active.dataset.version);
        select(versions[(index + 1) % versions.length]);
      };
      trigger.addEventListener("click", next);
      trigger.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          next();
        }
      });
    }
    const initial = panels.find(function (panel) {
      return panel.classList.contains("is-active");
    }) || panels[0];
    select(initial.dataset.version);
    const fromHash = window.location.hash.slice(1);
    if (versions.indexOf(fromHash) !== -1) select(fromHash);
  });
})();
"""


def _score_text(result):
    if result.benchmark in SNAKE_BENCHMARKS:
        lengths = result.details.get("seed_lengths")
        targets = result.details.get("target_lengths")
        if (
            isinstance(lengths, list)
            and isinstance(targets, list)
            and len(lengths) == len(targets)
            and targets
            and all(target > 0 for target in targets)
        ):
            percentages = [
                length / target * 100 for length, target in zip(lengths, targets, strict=True)
            ]
            return f"{statistics.fmean(percentages):.2f}% ± {statistics.pstdev(percentages):.2f}%"
        maximum = result.details.get("max_length", _SNAKE_MAX_LENGTH.get(result.benchmark, 64))
        mean = result.score / maximum * 100
        return f"{mean:.2f}% ± {result.score_std / maximum * 100:.2f}%"
    mean = result.score * 100
    if result.benchmark == "chess-eval":
        return f"{mean:.2f}% ± {result.score_std * 100:.2f}%"
    digits = 4 if not result.passed and mean >= 99.995 else 2
    return f"{mean:.{digits}f}%"


def _game_length_text(result):
    moves = result.details.get("seed_moves")
    return (
        "-"
        if not isinstance(moves, list) or not moves
        else f"{statistics.fmean(moves):.1f} ± {statistics.pstdev(moves):.1f}"
    )


def _plural(count, singular):
    return singular if count == 1 else f"{singular}s"


def _ranking_text(benchmark):
    if benchmark in SNAKE_BENCHMARKS:
        return "pass status, score, parameter count, then game length"
    return "pass status, score, then parameter count"


def _version_label(benchmark):
    if benchmark in _VERSION_LABELS:
        return _VERSION_LABELS[benchmark]
    return _SECTION_TITLES.get(benchmark, benchmark.replace("-", " ").title())


def _group_specs(results):
    present = {result.benchmark for result in results}
    groups = []
    covered = set()
    for spec in _GROUP_SPECS:
        versions = tuple(benchmark for benchmark in spec["benchmarks"] if benchmark in present)
        if not versions:
            continue
        groups.append({**spec, "benchmarks": versions})
        covered.update(versions)
    for benchmark in sorted(present - covered):
        groups.append(
            {
                "id": benchmark,
                "label": _version_label(benchmark),
                "eyebrow": "Other",
                "description": "",
                "benchmarks": (benchmark,),
                "accent": "default",
            }
        )
    return groups


def _group_info(benchmark):
    for spec in _GROUP_SPECS:
        if benchmark in spec["benchmarks"]:
            return spec["label"], spec["accent"]
    return _version_label(benchmark), "default"


def _submission_cell(result, link):
    name = html.escape(result.name)
    submission = html.escape(result.submission)
    secondary = (
        f'<span class="submission__id">{submission}</span>'
        if result.name != result.submission
        else ""
    )
    content = f'<span class="submission__name">{name}</span>{secondary}'
    if link:
        return f'<a href="{html.escape(link)}" aria-label="View details for {name}">{content}</a>'
    return content


def _table(results, submission_links=None):
    rows = []
    for rank, result in enumerate(results, 1):
        link = (submission_links or {}).get((result.benchmark, result.submission))
        status = "pass" if result.passed else "fail"
        rows.append(
            f'<tr><td class="rank">{rank}</td><td class="submission">{_submission_cell(result, link)}</td><td class="num score">{_score_text(result)}</td><td><span class="status status--{status}">{status}</span></td><td class="num">{result.parameter_count:,}</td></tr>'
        )
    label = _version_label(results[0].benchmark) if results else "Results"
    caption = (
        f"{html.escape(label)} results, ranked by {_ranking_text(results[0].benchmark)}."
        if results
        else "Results."
    )
    return f"""<table class="results"><caption class="sr-only">{caption}</caption><thead><tr><th scope="col" class="rank">#</th><th scope="col">Submission</th><th scope="col" class="num">Score</th><th scope="col">Status</th><th scope="col" class="num">Parameters</th></tr></thead><tbody>{"".join(rows)}</tbody></table>"""


def _detail_row(label, value):
    return f"<div><dt>{html.escape(label)}</dt><dd>{value}</dd></div>"


def _replay_text(payload):
    return "Available below" if payload is not None else "Not recorded"


def render_submission_html(result, payload, rank, total):
    """Render a submission detail page, embedding a replay when one is available."""
    group_label, accent = _group_info(result.benchmark)
    version_label = _version_label(result.benchmark)
    title = result.name or result.submission
    status = "pass" if result.passed else "fail"
    normalisation = result.details.get("normalisation")
    normalisation_text = str(normalisation).lower() if isinstance(normalisation, bool) else "-"
    rows = [
        _detail_row("Benchmark", html.escape(version_label)),
        _detail_row("Group", html.escape(group_label)),
        _detail_row("Submission", html.escape(result.submission)),
        _detail_row("Name", html.escape(result.name)),
        _detail_row("Author", html.escape(result.author)),
        _detail_row("Rank", f"#{rank} of {total}"),
        _detail_row(
            "Score", f'<span class="detail-score">{html.escape(_score_text(result))}</span>'
        ),
        _detail_row("Status", f'<span class="status status--{status}">{status}</span>'),
        _detail_row("Normalisation", html.escape(normalisation_text)),
        _detail_row("Parameters", f"{result.parameter_count:,}"),
    ]
    if result.benchmark in SNAKE_BENCHMARKS:
        rows.append(_detail_row("Game length", html.escape(_game_length_text(result))))
        seed_lengths = result.details.get("seed_lengths")
        if isinstance(seed_lengths, list) and seed_lengths:
            rows.append(_detail_row("Seed games", str(len(seed_lengths))))
        rows.append(_detail_row("Replay", html.escape(_replay_text(payload))))
    samples = result.details.get("samples")
    if isinstance(samples, int):
        rows.append(_detail_row("Samples", f"{samples:,}"))
    tolerance = result.details.get("tolerance_pawns")
    if isinstance(tolerance, (int, float)):
        rows.append(_detail_row("Tolerance", f"{tolerance:g} pawns"))
    meta = "".join(rows)

    if payload is not None:
        size = int(payload.get("size", 8))
        canvas = 40 * size
        script = (
            _GAME_SCRIPT.replace("__FRAMES__", json.dumps(payload["frames"]))
            .replace("__SIZE__", str(size))
            .replace("__BLOCKED__", json.dumps(payload.get("blocked", [])))
        )
        game_section = f"""<section class="detail-section detail-game" aria-labelledby="replay-title"><div class="detail-section-head"><h2 id="replay-title">Replay</h2><p>Seed {int(payload.get("seed", 0))} · {int(payload.get("moves", 0)):,} moves · final length {int(payload.get("length", 0))}</p></div><canvas id="board" width="{canvas}" height="{canvas}"></canvas><div class="controls"><button id="play" type="button">Play</button><button id="prev" type="button">Step back</button><button id="next" type="button">Step forward</button><label>Speed <select id="speed"><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select></label></div><p id="status" class="muted" aria-live="polite"></p></section>"""
        script_tag = f"<script>{script}</script>"
        grid_class = "detail-grid"
    else:
        game_section = ""
        script_tag = ""
        grid_class = "detail-grid detail-grid--single"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} — {html.escape(version_label)} — Minimal NN Competition</title><style>{_STYLE}{_DETAIL_STYLE}</style></head><body data-accent="{html.escape(accent)}"><div class="detail-shell"><header class="detail-head"><p class="eyebrow"><span class="detail-group">{html.escape(group_label)}</span> · {html.escape(version_label)}</p><h1 class="detail-title">{html.escape(title)}</h1><p class="detail-subtitle">Submission details</p><p class="detail-links"><a href="../">← Back to leaderboard</a><span>·</span><a href="../leaderboard.json">Raw leaderboard JSON</a></p></header><div class="{grid_class}">{game_section}<aside class="detail-panel" aria-labelledby="details-title"><h2 class="detail-kicker" id="details-title">Submission</h2><dl class="detail-meta">{meta}</dl></aside></div><footer class="detail-footer"><a href="../">Back to leaderboard</a><span>Minimal NN Competition</span></footer></div>{script_tag}</body></html>"""


def render_submission_pages(results, replays, output_dir):
    """Write one detail page per submission; return benchmark/submission links.

    Replay JSON is written under ``games/`` when a Snake replay is available.
    """
    pages_dir = output_dir / "submissions"
    games_dir = output_dir / "games"
    shutil.rmtree(pages_dir, ignore_errors=True)
    shutil.rmtree(games_dir, ignore_errors=True)
    totals = {}
    for result in results:
        totals[result.benchmark] = totals.get(result.benchmark, 0) + 1
    ranks = {}
    seen = {}
    links = {}
    for result in results:
        ranks[result.benchmark] = ranks.get(result.benchmark, 0) + 1
        stem = replay_stem(result.benchmark, result.submission)
        previous = seen.get(stem)
        if previous is not None and previous != (result.benchmark, result.submission):
            raise ValueError(
                f"Replay filename collision: {previous!r} and "
                f"{(result.benchmark, result.submission)!r} both map to {stem!r}"
            )
        seen[stem] = (result.benchmark, result.submission)
        payload = None
        if result.benchmark in SNAKE_BENCHMARKS:
            payload = replays.get(stem)
            if payload is not None:
                games_dir.mkdir(parents=True, exist_ok=True)
                (games_dir / f"{stem}.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
        pages_dir.mkdir(parents=True, exist_ok=True)
        (pages_dir / f"{stem}.html").write_text(
            render_submission_html(
                result, payload, ranks[result.benchmark], totals[result.benchmark]
            )
        )
        links[(result.benchmark, result.submission)] = f"submissions/{stem}.html"
    return links


def render_html(results, submission_links=None):
    groups = _group_specs(results)
    entries = {}
    for result in results:
        entries.setdefault(result.benchmark, []).append(result)
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    rail_links = []
    sections = []
    for group_number, spec in enumerate(groups, 1):
        versions = list(spec["benchmarks"])
        if not versions:
            continue
        group_id = spec["id"]
        accent = spec["accent"]
        label = spec["label"]
        default_benchmark = versions[0]
        default_label = _version_label(default_benchmark)
        group_count = sum(len(entries[benchmark]) for benchmark in versions)
        group_pass = sum(
            sum(1 for result in entries[benchmark] if result.passed) for benchmark in versions
        )
        rail_links.append(
            f'<a href="#{html.escape(group_id)}" data-accent="{html.escape(accent)}"><span class="rail__dot" aria-hidden="true"></span><span class="rail__index">{group_number:02d}</span><strong>{html.escape(label)}</strong><small>{group_count} {_plural(group_count, "run")}</small></a>'
        )
        if len(versions) > 1:
            title = (
                f'<span class="group__title-trigger" data-cycle-trigger>'
                f'<span class="group__title-text">{html.escape(label)}</span>'
                f'<span class="cycle" aria-hidden="true">↻</span></span>'
            )
        else:
            title = f'<span class="group__title-text">{html.escape(label)}</span>'
        description = (
            f'<p class="group__desc">{html.escape(spec["description"])}</p>'
            if spec["description"]
            else ""
        )
        stats = (
            f'<div class="group__stats"><span><strong data-count>{group_count}</strong> '
            f"<span data-count-word>{_plural(group_count, 'run')}</span></span>"
            f"<span><strong data-pass>{group_pass}</strong> pass</span></div>"
        )
        version_buttons = []
        for benchmark in versions:
            version_label = _version_label(benchmark)
            active = benchmark == default_benchmark
            version_buttons.append(
                f'<button type="button" class="version-link{" is-active" if active else ""}" data-version-button data-version="{html.escape(benchmark)}" aria-controls="panel-{html.escape(benchmark)}" aria-pressed="{"true" if active else "false"}">{html.escape(version_label)}</button>'
            )
        if len(versions) > 1:
            version_controls = (
                f'<div class="version-links" role="group" aria-label="{html.escape(label)} versions">'
                f"{''.join(version_buttons)}</div>"
            )
            hint = (
                f'<p class="version-hint"><span data-current aria-live="polite">'
                f"{html.escape(default_label)} · 1 of {len(versions)}</span> · click the title to cycle</p>"
            )
        else:
            version_controls = ""
            hint = f'<p class="version-hint"><span>{html.escape(default_label)} · 1 of 1</span></p>'
        version_row = (
            f'<div class="version-row"><p class="version-row__label">Version</p>'
            f"{version_controls}{hint}</div>"
        )
        panels = []
        for benchmark in versions:
            version_entries = entries[benchmark]
            version_label = _version_label(benchmark)
            count = len(version_entries)
            pass_count = sum(1 for result in version_entries if result.passed)
            active = benchmark == default_benchmark
            note = f"<strong>{html.escape(version_label)}</strong>"
            detail = _PANEL_NOTES.get(benchmark)
            if detail:
                note += f" · {html.escape(detail)}"
            note += f" · Ranked by {_ranking_text(benchmark)}."
            note += " · Select a submission for full details."
            table = _table(version_entries, submission_links)
            panels.append(
                f'<div class="version-panel{" is-active" if active else ""}" id="panel-{html.escape(benchmark)}" data-version="{html.escape(benchmark)}" data-label="{html.escape(version_label)}" data-count="{count}" data-pass="{pass_count}">'
                f'<p class="table-note">{note}</p>'
                f'<div class="table-wrap" role="region" tabindex="0" aria-label="{html.escape(version_label)} results">{table}</div>'
                f"</div>"
            )
        sections.append(
            f'<section class="group" id="{html.escape(group_id)}" data-group-section data-accent="{html.escape(accent)}" data-title="{html.escape(label)}" aria-labelledby="{html.escape(group_id)}-title">'
            f'<div class="group__head"><div><p class="eyebrow"><span class="group__index">{group_number:02d}</span> / {html.escape(spec["eyebrow"])}</p>'
            f'<h2 class="group__title" id="{html.escape(group_id)}-title">{title}</h2>{description}</div>{stats}</div>'
            f"{version_row}{''.join(panels)}</section>"
        )
    rail_note = (
        f"{total} {_plural(total, 'run')} · {passed} pass<br>"
        f'<span class="js-only">Click a group title to cycle its versions.</span>'
        if groups
        else ""
    )
    content = "".join(sections) or "<p>No results have been recorded yet.</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Minimal NN Competition</title><script>document.documentElement.classList.add("js");</script><style>{_STYLE}{_SITE_STYLE}</style></head><body><div class="shell"><aside class="rail"><p class="brand-mark"><span class="brand-dots" aria-hidden="true"><span></span><span></span><span></span></span>Minimal NN Competition</p><p class="rail__label">Benchmark groups</p><nav aria-label="Benchmark groups">{"".join(rail_links)}</nav><p class="rail__note">{rail_note}</p></aside><main class="main"><header class="masthead"><p class="eyebrow">Public leaderboard</p><h1>Minimal NN Competition</h1><p class="lead">Small PyTorch state-dict submissions for image classification, Snake variants, and chess evaluation.</p><p class="links"><a href="https://github.com/wd7512/minimal-nn-competition/blob/main/CONTRIBUTING.md">Submit a model</a><span>·</span><a href="leaderboard.json">Raw leaderboard JSON</a></p></header>{content}<footer class="footer"><p>Minimal NN Competition · static results</p><a class="version" href="internal/speed/" title="Hidden Snake Simple 8x8 environment speed leaderboard" aria-label="Hidden Snake Simple 8x8 environment speed leaderboard">v{html.escape(__version__)}</a></footer></main></div><script>{_SITE_SCRIPT}</script></body></html>
"""


def render_site(results, output_dir, replays=None):
    ranked = rank_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(
        json.dumps([asdict(result) for result in ranked], indent=2, sort_keys=True) + "\n"
    )
    submission_links = render_submission_pages(ranked, replays or {}, output_dir)
    index = output_dir / "index.html"
    index.write_text(render_html(ranked, submission_links or None))
    (output_dir / ".nojekyll").write_text("")
    return index
