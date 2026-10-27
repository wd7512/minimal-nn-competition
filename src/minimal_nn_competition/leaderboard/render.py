"""Render ranked results as a static GitHub Pages site."""

import html
import json
import shutil
import statistics
from dataclasses import asdict
from datetime import date

from minimal_nn_competition import __version__
from minimal_nn_competition.benchmarks.common.control.replay import steps_text
from minimal_nn_competition.benchmarks.common.snake.replay import replay_stem
from minimal_nn_competition.benchmarks.registry import (
    CHESS_BENCHMARKS,
    IMAGE_BENCHMARKS,
    RL_BENCHMARKS,
    SNAKE_BENCHMARKS,
)
from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.leaderboard.theme import SITE_BASE_CSS, SITE_THEME_CSS

_GROUP_SPECS = (
    {
        "id": "snake",
        "label": "Snake",
        "benchmarks": SNAKE_BENCHMARKS,
        "accent": "snake",
    },
    {
        "id": "image-compression",
        "label": "Image compression",
        "benchmarks": IMAGE_BENCHMARKS,
        "accent": "image-compression",
    },
    {
        "id": "chess",
        "label": "Chess evaluation",
        "benchmarks": CHESS_BENCHMARKS,
        "accent": "chess",
    },
    {
        "id": "control",
        "label": "Control",
        "benchmarks": RL_BENCHMARKS,
        "accent": "control",
    },
)
BENCHMARK_GROUPS = tuple(spec["id"] for spec in _GROUP_SPECS)
GROUP_NAMES = BENCHMARK_GROUPS

_VERSION_LABELS = {
    "snake-simple-8x8": "Simple 8×8",
    "snake-simple-16x16": "Simple 16×16",
    "snake-obstacles-8x8": "Obstacles 8×8",
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "mnist-shuffled": "MNIST Shuffled",
    "fashion-mnist-shuffled": "Fashion-MNIST shuffled",
    "chess-eval-10k": "Stockfish 10k",
    "chess-eval-100k": "Stockfish 100k",
    "cartpole": "CartPole",
    "pendulum": "Discrete torque",
    "lunarlander": "Lunar touchdown",
}
_STYLE = SITE_THEME_CSS + SITE_BASE_CSS + """
[data-accent="snake"] { --accent: var(--accent-snake); }
[data-accent="image-compression"] { --accent: var(--accent-image-compression); }
[data-accent="chess"] { --accent: var(--accent-chess); }
[data-accent="control"] { --accent: var(--accent-control); }
[data-accent="default"] { --accent: var(--accent-snake); }
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font: 700 10px/1 var(--mono);
  letter-spacing: .08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.status::before { content: ""; width: 6px; height: 6px; background: currentColor; flex: none; }
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
  font: 700 16px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--ink);
}
.brand-dots { display: inline-flex; gap: 3px; }
.brand-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: conic-gradient(var(--accent) var(--pass-percent), var(--rule) 0);
}
.rail__label {
  margin: 24px 0 8px;
  font: 700 15px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
.rail nav { display: grid; border-top: 1px solid var(--rule); }
.rail a {
  display: grid;
  grid-template-columns: 6px 22px minmax(0, 1fr);
  gap: 2px 8px;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--rule);
  color: var(--ink);
  text-decoration: none;
}
.rail__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: conic-gradient(var(--accent) var(--pass-percent), var(--rule) 0);
}
.rail__index { font: 10px/1.4 var(--mono); color: var(--muted); }
.rail a strong { font-weight: 600; }
.rail a:hover strong { text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: 3px; }
.rail a small { grid-column: 3; font: 10px/1.4 var(--mono); color: var(--muted); white-space: nowrap; }
.rail__note { margin: 20px 0 0; font: 11px/1.5 var(--mono); color: var(--muted); }
.main { min-width: 0; }
.masthead { border-bottom: 2px solid var(--accent-cta); padding-bottom: 36px; }
.masthead .eyebrow {
  display: inline-flex;
  padding: 6px 10px;
  background: var(--accent-cta);
  color: var(--accent-cta-ink);
  font-size: 10px;
  letter-spacing: .1em;
}
.eyebrow {
  margin: 0;
  font: 700 15px/1.4 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
}
h1 {
  margin: 20px 0 0;
  font: 400 clamp(3rem, 6vw, 5.4rem)/.96 var(--display);
  letter-spacing: -.04em;
  max-width: 14ch;
}
.lead { margin: 20px 0 0; max-width: 54ch; color: var(--ink); font-size: 19px; }
.links {
  margin: 24px 0 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 14px 20px;
  font: 12px/1.4 var(--mono);
}
.links .primary-action {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  border: 2px solid var(--accent-cta-ink);
  border-radius: 3px;
  background: var(--accent-cta);
  box-shadow: 3px 3px 0 var(--accent-cta-ink);
  color: var(--accent-cta-ink);
  font-weight: 700;
  text-decoration: none;
  transition: transform 150ms ease-out, box-shadow 150ms ease-out, background-color 150ms ease-out;
}
.links .primary-action:hover {
  transform: translate(2px, 2px);
  box-shadow: 1px 1px 0 var(--accent-cta-ink);
  background: var(--accent-cta-hover);
  color: var(--accent-cta-ink);
  text-decoration: none;
}
.links .primary-action:active {
  transform: translate(3px, 3px);
  box-shadow: none;
}
.featured {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 24px;
  margin: 28px 0 8px;
  padding: 20px 24px;
  border: 1px solid var(--rule-strong);
  border-top: 4px solid var(--accent-cta);
  background: var(--surface);
}
.featured__intro h2 { margin: 4px 0 0; font: 400 1.55rem/1.15 var(--display); }
.featured__intro h2 a { color: var(--ink); text-decoration-color: var(--accent-cta); text-underline-offset: 4px; }
.featured__stats { display: flex; gap: 24px; margin: 0; }
.featured__stat { margin: 0; }
.featured__stat strong { display: block; font: 700 16px/1.2 var(--mono); white-space: nowrap; }
.featured__stat span { display: block; margin-top: 4px; color: var(--muted); font: 10px/1.4 var(--mono); }
.featured__link {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--rule-strong);
  color: var(--ink);
  font: 11px/1.4 var(--mono);
  text-decoration: none;
  white-space: nowrap;
}
.featured__link:hover { border-color: var(--accent-cta-ink); background: var(--accent-cta); color: var(--accent-cta-ink); }
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
.version-panel { margin-top: 20px; }
.version-panel + .version-panel { margin-top: 40px; }
.js .version-panel:not(.is-active) { display: none; }
.passed-toggle { display: none; margin: 0 0 8px; font: 11px/1.4 var(--mono); color: var(--muted); }
.js .passed-toggle { display: flex; align-items: center; gap: 8px; }
.passed-toggle input { width: 14px; height: 14px; accent-color: var(--pass); }
.js .version-panel.hide-failed tbody tr.is-fail { display: none; }
.table-wrap {
  overflow-x: auto;
  border-top: 1px solid var(--rule-strong);
  border-bottom: 1px solid var(--rule-strong);
  -webkit-overflow-scrolling: touch;
}
.empty-state {
  margin: 0;
  padding: 16px 12px;
  border-block: 1px solid var(--rule-strong);
  color: var(--muted);
}
table.results { width: 100%; border-collapse: collapse; min-width: 30rem; }
.results th, .results td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--rule);
  text-align: left;
  vertical-align: top;
}
.results tbody tr.is-pass { background: var(--pass-tint); }
.results tbody tr.is-fail { background: var(--fail-tint); }
.results tbody tr.is-pass td:first-child { background: var(--pass-tint); }
.results tbody tr.is-fail td:first-child { background: var(--fail-tint); }
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
.submission__name { font-weight: 400; font-size: 12px; }
.submission__id { font: 10px/1.35 var(--mono); color: var(--muted); }
.submission__meta { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }
.submission__meta .score { font-weight: 600; color: var(--ink); font-size: 11px; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.num { font-family: var(--mono); font-size: 12px; }
td.date { white-space: nowrap; font: 11px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
td.score { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--ink); }
.std { color: var(--muted); font-weight: 400; font-size: 11px; white-space: nowrap; }
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
.version { color: inherit; opacity: .9; text-decoration: underline; text-underline-offset: 3px; }
.version:hover, .version:focus-visible { opacity: 1; }
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
  .rail__note { display: block; margin-top: 12px; }
}
@media (max-width: 700px) {
  .shell { padding: 24px 16px 56px; }
  .rail nav { grid-template-columns: 1fr; }
  .rail a { border-left: 0; border-bottom: 1px solid var(--rule); padding-left: 0; }
  .rail a + a { border-left: 0; padding-left: 0; }
  .group__head { grid-template-columns: 1fr; }
  .featured { grid-template-columns: 1fr; gap: 16px; padding: 16px; }
  .featured__stats { gap: 20px; }
  .results th, .results td { padding: 8px 8px; }
  td.submission a { gap: 1px; }
  .group { padding: 36px 0 20px; }
  .table-wrap { overflow-x: visible; }
  table.results { min-width: 0; }
  td.num, th.num { white-space: normal; }
  .std { display: inline; white-space: nowrap; }
  .results th:first-child, .results td:first-child { position: static; }
  .footer { flex-direction: column; gap: 8px; }
}
"""

_DETAIL_STYLE = """
.detail-shell { width: 100%; max-width: 1120px; margin: 0 auto; padding: 48px 32px 80px; }
.detail-head { border-bottom: 1px solid var(--rule-strong); padding-bottom: 24px; }
.eyebrow {
  margin: 0;
  font: 700 15px/1.4 var(--mono);
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
  background: var(--surface);
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
  font: 700 15px/1.4 var(--mono);
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
const themeColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
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
    ctx.fillStyle = themeColor("--chart-obstacle");
    ctx.fillRect(blockedCell[0] * cell, blockedCell[1] * cell, cell, cell);
  });
  const food = FRAMES[frame].food;
  if (food) {
    ctx.fillStyle = themeColor("--chart-danger");
    ctx.fillRect(food[0] * cell, food[1] * cell, cell, cell);
  }
  const snake = FRAMES[frame].snake;
  snake.forEach((segment, index) => {
    ctx.fillStyle = index === 0
      ? themeColor("--chart-snake-head")
      : themeColor("--chart-snake-body");
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
  }, Math.max(1, Math.floor(240 / Number(speed.value))));
}
playButton.addEventListener("click", play);
document.getElementById("prev").addEventListener("click", () => { stop(); step(-1); });
document.getElementById("next").addEventListener("click", () => { stop(); step(1); });
speed.addEventListener("change", () => { if (playing) { stop(); play(); } });
draw();
"""

_CONTROL_GAME_SCRIPT = """
const FRAMES = __FRAMES__;
const KIND = "__KIND__";
const OUTCOME = "__OUTCOME__";
const TORQUES = [-2.0, -1.0, 0.0, 1.0, 2.0];
const THRUSTS = ["coast", "main", "left", "right"];
const themeColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const playButton = document.getElementById("play");
const speed = document.getElementById("speed");
let frame = 0;
let playing = false;
let timer = null;
function last() {
  return FRAMES.length - 1;
}
function deg(radians) {
  return (Number(radians) * 180 / Math.PI).toFixed(1);
}
function drawCartpole(state) {
  const rail = canvas.height - 80;
  const toX = (x) => 24 + ((x + 2.4) / 4.8) * (canvas.width - 48);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = themeColor("--chart-boundary");
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(16, rail);
  ctx.lineTo(canvas.width - 16, rail);
  ctx.stroke();
  ctx.strokeStyle = themeColor("--fail");
  ctx.setLineDash([4, 3]);
  [toX(-2.4), toX(2.4)].forEach((lineX) => {
    ctx.beginPath();
    ctx.moveTo(lineX, rail - 110);
    ctx.lineTo(lineX, rail + 14);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  const cartX = Math.max(24, Math.min(canvas.width - 24, toX(state[0])));
  ctx.fillStyle = themeColor("--chart-cart");
  ctx.fillRect(cartX - 24, rail - 28, 48, 28);
  const pivotX = cartX;
  const pivotY = rail - 28;
  const tipX = pivotX + 200 * Math.sin(state[2]);
  const tipY = pivotY - 200 * Math.cos(state[2]);
  ctx.strokeStyle = themeColor("--accent-image-compression");
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pivotX, pivotY);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();
  ctx.lineCap = "butt";
  ctx.fillStyle = themeColor("--accent-image-compression");
  ctx.beginPath();
  ctx.arc(tipX, tipY, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = themeColor("--accent-chess");
  ctx.beginPath();
  ctx.arc(pivotX, pivotY, 5, 0, 2 * Math.PI);
  ctx.fill();
  let text = "step " + frame + "/" + last()
    + ", x " + Number(state[0]).toFixed(2)
    + ", θ " + deg(state[2]) + " degrees";
  if (frame === last()) {
    text += " · " + OUTCOME;
  }
  status.textContent = text;
}
function drawPendulum(state, action) {
  const pivotX = canvas.width / 2;
  const pivotY = canvas.height * 0.47;
  const arm = canvas.height * 0.34;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const tipX = pivotX + arm * Math.sin(state[0]);
  const tipY = pivotY - arm * Math.cos(state[0]);
  ctx.strokeStyle = themeColor("--chart-danger");
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pivotX, pivotY);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();
  ctx.lineCap = "butt";
  ctx.fillStyle = themeColor("--chart-danger");
  ctx.beginPath();
  ctx.arc(pivotX, pivotY, 10, 0, 2 * Math.PI);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(tipX, tipY, 10, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = themeColor("--chart-cart");
  ctx.beginPath();
  ctx.arc(pivotX, pivotY, 4, 0, 2 * Math.PI);
  ctx.fill();
  const torque = action === null ? 0 : TORQUES[action];
  if (action !== null && torque !== 0) {
    const R = 50;
    const span = 100 * Math.PI / 180;
    const base = -Math.PI / 2;
    const a0 = base;
    const a1 = torque > 0 ? base + span : base - span;
    ctx.strokeStyle = themeColor("--chart-muted");
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, R, a0, a1, torque < 0);
    ctx.stroke();
    const ex = pivotX + R * Math.cos(a1);
    const ey = pivotY + R * Math.sin(a1);
    const tAng = a1 + (torque > 0 ? Math.PI / 2 : -Math.PI / 2);
    const ux = Math.cos(tAng);
    const uy = Math.sin(tAng);
    const px = -uy;
    const py = ux;
    const bx = ex - ux * 14;
    const by = ey - uy * 14;
    ctx.fillStyle = themeColor("--chart-muted");
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(bx + px * 7, by + py * 7);
    ctx.lineTo(bx - px * 7, by - py * 7);
    ctx.closePath();
    ctx.fill();
  }
  let text = "step " + frame + "/" + last()
    + ", θ " + deg(state[0]) + " degrees"
    + ", ω " + Number(state[1]).toFixed(2);
  if (frame === last()) {
    text += " · " + OUTCOME;
  }
  status.textContent = text;
}
function drawLunarlander(state, action) {
  const ground = canvas.height - 40;
  const toX = (x) => 30 + ((x + 1) / 2) * (canvas.width - 60);
  const toY = (y) => ground - y * ((ground - 100) / 1.5);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = themeColor("--chart-cart");
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, ground);
  ctx.lineTo(canvas.width, ground);
  ctx.stroke();
  ctx.fillStyle = themeColor("--chart-snake-head");
  ctx.fillRect(toX(-0.25), ground - 6, toX(0.25) - toX(-0.25), 6);
  [toX(-0.25), toX(0.25)].forEach((edge, idx) => {
    ctx.strokeStyle = themeColor("--chart-cart");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(edge, ground - 6);
    ctx.lineTo(edge, ground - 36);
    ctx.stroke();
    ctx.fillStyle = themeColor("--accent-image-compression");
    ctx.beginPath();
    if (idx === 0) {
      ctx.moveTo(edge, ground - 36);
      ctx.lineTo(edge, ground - 48);
      ctx.lineTo(edge + 16, ground - 42);
    } else {
      ctx.moveTo(edge, ground - 36);
      ctx.lineTo(edge, ground - 48);
      ctx.lineTo(edge - 16, ground - 42);
    }
    ctx.closePath();
    ctx.fill();
  });
  for (let index = Math.max(0, frame - 24); index < frame; index += 1) {
    ctx.globalAlpha = 0.45;
    ctx.fillStyle = themeColor("--accent-chess");
    ctx.beginPath();
    ctx.arc(toX(FRAMES[index].state[0]), toY(FRAMES[index].state[1]), 3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  const px = toX(state[0]);
  const py = Math.min(toY(state[1]), ground + 4);
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(Number(state[4]));
  ctx.fillStyle = themeColor("--accent-control");
  ctx.strokeStyle = themeColor("--chart-cart");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, -13);
  ctx.lineTo(9, -4);
  ctx.lineTo(9, 10);
  ctx.lineTo(-9, 10);
  ctx.lineTo(-9, -4);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = themeColor("--chart-obstacle");
  if (action === 1) {
    ctx.beginPath();
    ctx.arc(0, 16, 4 + (frame % 3), 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, 24, 3, 0, 2 * Math.PI);
    ctx.fill();
  } else if (action === 2) {
    ctx.beginPath();
    ctx.arc(-14, 5, 3 + (frame % 2), 0, 2 * Math.PI);
    ctx.fill();
  } else if (action === 3) {
    ctx.beginPath();
    ctx.arc(14, 5, 3 + (frame % 2), 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
  let text = "step " + frame + "/" + last()
    + ", x " + Number(state[0]).toFixed(2)
    + ", y " + Number(state[1]).toFixed(2)
    + ", thrust " + (action === null ? "—" : THRUSTS[action]);
  if (frame === last() && OUTCOME) {
    text += " · " + OUTCOME;
  }
  status.textContent = text;
}
function draw() {
  const current = FRAMES[frame];
  if (KIND === "cartpole") {
    drawCartpole(current.state);
  } else if (KIND === "pendulum") {
    drawPendulum(current.state, current.action);
  } else {
    drawLunarlander(current.state, current.action);
  }
}
function step(delta) {
  frame = Math.min(last(), Math.max(0, frame + delta));
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
  if (frame >= last()) {
    frame = 0;
  }
  playing = true;
  playButton.textContent = "Pause";
  timer = setInterval(() => {
    frame += 1;
    if (frame >= last()) {
      frame = last();
      stop();
    }
    draw();
  }, Math.max(1, Math.floor(240 / Number(speed.value))));
}
  playButton.addEventListener("click", play);
  document.getElementById("prev").addEventListener("click", () => { stop(); step(-1); });
  document.getElementById("next").addEventListener("click", () => { stop(); step(1); });
  speed.addEventListener("change", () => { if (playing) { stop(); play(); } });
  draw();
"""

_REPLAY_BOX = 480

_CONTROL_GAME_LABEL = {
    "cartpole": "CartPole replay",
    "pendulum": "Pendulum replay",
    "lunarlander": "LunarLander replay",
}


def _control_game_section(kind, payload):
    seed = int(payload.get("seed", 0))
    steps = int(payload.get("steps", 0))
    total = float(payload.get("total_reward", 0.0))
    terminated = bool(payload.get("terminated", False))
    if kind == "cartpole":
        outcome = "fell" if terminated else "balanced"
    elif kind == "pendulum":
        outcome = f"total return {total:.2f}"
    elif payload.get("landed"):
        outcome = "landed"
    elif terminated:
        outcome = "crashed"
    else:
        outcome = "timed out"
    width, height = _REPLAY_BOX, _REPLAY_BOX
    label = _CONTROL_GAME_LABEL[kind]
    caption = f"Seed {seed} · {steps:,} steps · {outcome}"
    canvas_label = (
        f"{label}. Seed {seed}, {steps:,} steps, {outcome}. "
        "Use Play and Step buttons below; current position is announced in status text."
    )
    canvas_fallback = (
        f"{label}: Seed {seed}, {steps:,} steps, {outcome}. Enable JavaScript to play."
    )
    script = (
        _CONTROL_GAME_SCRIPT.replace("__FRAMES__", json.dumps(payload["frames"]))
        .replace("__KIND__", kind)
        .replace("__OUTCOME__", outcome)
    )
    section = (
        '<section class="detail-section detail-game" aria-labelledby="replay-title">'
        '<div class="detail-section-head"><h2 id="replay-title">Replay</h2>'
        f"<p>{html.escape(caption)}</p></div>"
        f'<canvas id="board" width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(canvas_label)}" aria-describedby="status">{html.escape(canvas_fallback)}</canvas>'
        '<div class="controls"><button id="play" type="button">Play</button>'
        '<button id="prev" type="button">Step back</button>'
        '<button id="next" type="button">Step forward</button>'
        f"{_speed_select('4')}</div>"
        '<noscript><p class="muted">Replay needs JavaScript. Summary above gives seed, steps and outcome.</p></noscript>'
        '<p id="status" class="muted" aria-live="polite"></p></section>'
    )
    return section, script


_SITE_SCRIPT = """
(function () {
  const groups = Array.from(document.querySelectorAll("[data-group-section]"));
  groups.forEach(function (group) {
    const panels = Array.from(group.querySelectorAll(".version-panel"));
    if (!panels.length) return;
    const versions = panels.map(function (panel) { return panel.dataset.version; });
    const links = Array.from(group.querySelectorAll("[data-version-button]"));
    const current = group.querySelector("[data-current]");
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
    }
    links.forEach(function (link) {
      link.addEventListener("click", function () { select(link.dataset.version); });
    });
    panels.forEach(function (panel) {
      const toggle = panel.querySelector("[data-passed-toggle]");
      if (!toggle) return;
      const body = panel.querySelector("tbody");
      const emptyRow = body && body.querySelector("[data-empty-row]");
      const submissions = body
        ? Array.from(body.querySelectorAll("tr.is-pass, tr.is-fail"))
        : [];
      const apply = function () {
        panel.classList.toggle("hide-failed", toggle.checked);
        if (!emptyRow) return;
        const hasVisibleResults = submissions.some(function (row) {
          return !toggle.checked || !row.classList.contains("is-fail");
        });
        emptyRow.hidden = hasVisibleResults;
        if (!hasVisibleResults) {
          const message = emptyRow.querySelector("[data-empty-message]");
          if (message) {
            message.textContent = toggle.checked
              ? "Nobody’s found a passing solution for this one yet — maybe you can?"
              : "No submissions yet — want to be the first?";
          }
        }
      };
      toggle.addEventListener("change", apply);
      apply();
    });
    const initial = panels.find(function (panel) {
      return panel.classList.contains("is-active");
    }) || panels[0];
    select(initial.dataset.version);
    let fromHash = window.location.hash.slice(1);
    if (fromHash.indexOf("panel-") === 0) fromHash = fromHash.slice(6);
    if (versions.indexOf(fromHash) !== -1) {
      select(fromHash);
      if (!document.getElementById(window.location.hash.slice(1))) {
        try { group.scrollIntoView(); } catch (e) {}
      }
    }
  });
})();
"""


def _score_text(result):
    # Score is the fraction of seeds or samples that pass for every benchmark.
    percent = result.score * 100
    digits = 4 if not result.passed and percent >= 99.995 else 2
    return f"{percent:.{digits}f}%"


def _game_length_text(result):
    moves = result.details.get("seed_moves")
    return (
        "-"
        if not isinstance(moves, list) or not moves
        else f"{statistics.fmean(moves):.1f} ± {statistics.pstdev(moves):.1f}"
    )


def _error_text(result):
    mae = result.details.get("mae")
    if not isinstance(mae, (int, float)):
        return "-"
    return f"{mae:.4f}"


def _reward_text(result):
    if result.benchmark == "cartpole":
        return steps_text(result)
    mean_return = result.details.get("mean_return")
    if not isinstance(mean_return, (int, float)):
        return "-"
    return f"{mean_return:.2f}"


def _fourth_text(result):
    if result.benchmark in SNAKE_BENCHMARKS:
        return _game_length_text(result)
    if result.benchmark in CHESS_BENCHMARKS:
        return _error_text(result)
    if result.benchmark in RL_BENCHMARKS:
        return _reward_text(result)
    return "-"


def _fourth_html(result):
    text = _fourth_text(result)
    if "±" in text:
        main, std = text.split("±", 1)
        return f'{html.escape(main.strip())} <span class="std">± {html.escape(std.strip())}</span>'
    return html.escape(text)


def _fourth_header(benchmark):
    if benchmark in SNAKE_BENCHMARKS:
        return "Game length"
    if benchmark in CHESS_BENCHMARKS:
        return "Error"
    if benchmark in RL_BENCHMARKS:
        return "Reward"
    return None


def _plural(count, singular):
    return singular if count == 1 else f"{singular}s"


def _ranking_text(benchmark):
    if benchmark in SNAKE_BENCHMARKS:
        return (
            "pass status, score, parameter count, then game length; "
            "failed submissions rank by longest game first"
        )
    if benchmark in CHESS_BENCHMARKS:
        return "pass status, score, parameter count, then mean error"
    return "pass status, score, then parameter count"


def _version_label(benchmark):
    if benchmark in _VERSION_LABELS:
        return _VERSION_LABELS[benchmark]
    return benchmark.replace("-", " ").title()


def _group_specs(results):
    present = {result.benchmark for result in results}
    groups = []
    covered = set()
    for spec in _GROUP_SPECS:
        if spec["id"] == "image-compression" and any(
            benchmark in present for benchmark in IMAGE_BENCHMARKS
        ):
            versions = IMAGE_BENCHMARKS
        else:
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


def _pass_percent(passed, total):
    return f"{100 * passed / total:.1f}%" if total else "0%"


def _submission_cell(result, link, status=None, score_html=None):
    name = html.escape(result.name)
    submission = html.escape(result.submission)
    secondary = (
        f'<span class="submission__id">{submission}</span>'
        if result.name != result.submission
        else ""
    )
    meta = ""
    if score_html:
        meta = f'<span class="submission__meta"><span class="score">{score_html}</span></span>'
    content = f'<span class="submission__name">{name}</span>{secondary}{meta}'
    if link:
        label = (
            f"View details for {result.name}, {status}"
            if status
            else f"View details for {result.name}"
        )
        return f'<a href="{html.escape(link)}" aria-label="{html.escape(label)}">{content}</a>'
    return content


def _submission_guide_url(benchmark):
    if benchmark in SNAKE_BENCHMARKS:
        guide = "how-to-make-a-snake-submission.md"
    elif benchmark in IMAGE_BENCHMARKS:
        guide = "how-to-make-an-image-compression-submission.md"
    elif benchmark in CHESS_BENCHMARKS:
        guide = "how-to-make-a-chess-submission.md"
    elif benchmark in RL_BENCHMARKS:
        guide = "how-to-make-a-control-submission.md"
    else:
        return "https://github.com/wd7512/minimal-nn-competition/blob/main/CONTRIBUTING.md"
    return (
        "https://github.com/wd7512/minimal-nn-competition/blob/main/how-to/" + guide
    )


def _submission_date_html(value):
    if not value:
        return "—"
    try:
        submitted = date.fromisoformat(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    day = submitted.day
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    label = f"{day}{suffix} {submitted.strftime('%b')} '{submitted.strftime('%y')}"
    return f'<time datetime="{submitted.isoformat()}">{label}</time>'


def _table(results, submission_links=None, benchmark=None):
    rows = []
    benchmark = results[0].benchmark if results else benchmark
    fourth = _fourth_header(benchmark) if benchmark else None
    for rank, result in enumerate(results, 1):
        link = (submission_links or {}).get((result.benchmark, result.submission))
        status = "pass" if result.passed else "fail"
        row_class = "is-pass" if result.passed else "is-fail"
        cells = (
            f'<td class="rank">{rank}</td>'
            f'<td class="submission">{_submission_cell(result, link, status, html.escape(_score_text(result)))}</td>'
            f'<td class="date">{_submission_date_html(result.submitted_at)}</td>'
            f'<td class="num">{result.parameter_count:,}</td>'
        )
        if fourth is not None:
            cells += f'<td class="num">{_fourth_html(result)}</td>'
        rows.append(f'<tr class="{row_class}">{cells}</tr>')
    label = _version_label(benchmark) if benchmark else "Results"
    fourth_header = (
        f'<th scope="col" class="num">{html.escape(fourth)}</th>' if fourth is not None else ""
    )
    caption = (
        f"{html.escape(label)} results, ranked by {_ranking_text(benchmark)}."
        if benchmark
        else "Results."
    )
    if benchmark:
        initially_hidden = " hidden" if results else ""
        empty_row = (
            f'<tr class="empty-row" data-empty-row{initially_hidden}>'
            f'<td colspan="{4 + int(fourth is not None)}">'
            '<span data-empty-message>Nobody’s found a passing solution for this one yet — '
            'maybe you can?</span> '
            f'<a href="{html.escape(_submission_guide_url(benchmark))}">How to submit</a>'
            "</td></tr>"
        )
        rows.append(empty_row)
    return f"""<table class="results"><caption class="sr-only">{caption}</caption><thead><tr><th scope="col" class="rank">#</th><th scope="col">Name</th><th scope="col" class="date">Submitted</th><th scope="col" class="num">Parameters</th>{fourth_header}</tr></thead><tbody>{"".join(rows)}</tbody></table>"""


def _detail_row(label, value):
    return f"<div><dt>{html.escape(label)}</dt><dd>{value}</dd></div>"


def _replay_text(payload):
    return "Available below" if payload is not None else "Not recorded"


_SPEED_OPTIONS = ("1", "2", "4", "8", "16", "32", "64")


def _speed_select(default):
    options = "".join(
        f'<option value="{value}"{" selected" if value == default else ""}>{value}x</option>'
        for value in _SPEED_OPTIONS
    )
    return f'<label>Speed <select id="speed">{options}</select></label>'


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
    ]
    if result.name != result.submission:
        rows.append(_detail_row("Name", html.escape(result.name)))
    rows.extend(
        [
            _detail_row("Author", html.escape(result.author)),
            _detail_row("Rank", f"#{rank} of {total}"),
            _detail_row(
                "Score", f'<span class="detail-score">{html.escape(_score_text(result))}</span>'
            ),
        ]
    )
    rows.extend(
        [
            _detail_row("Status", f'<span class="status status--{status}">{status}</span>'),
            _detail_row("Normalisation", html.escape(normalisation_text)),
        ]
    )
    structure = result.details.get("structure")
    if isinstance(structure, list) and structure:
        rows.append(_detail_row("Structure", f"[{','.join(str(int(size)) for size in structure)}]"))
    rows.append(_detail_row("Parameters", f"{result.parameter_count:,}"))
    if result.benchmark in SNAKE_BENCHMARKS:
        rows.append(_detail_row("Game length", html.escape(_game_length_text(result))))
        seed_lengths = result.details.get("seed_lengths")
        if isinstance(seed_lengths, list) and seed_lengths:
            rows.append(_detail_row("Seed games", str(len(seed_lengths))))
        rows.append(_detail_row("Replay", html.escape(_replay_text(payload))))
    if result.benchmark in RL_BENCHMARKS:
        rows.append(_detail_row("Reward", html.escape(_reward_text(result))))
        games = result.details.get("seed_steps")
        if not isinstance(games, list):
            games = result.details.get("seed_returns")
        if isinstance(games, list):
            rows.append(_detail_row("Seed games", str(len(games))))
        if result.benchmark == "lunarlander":
            landed = result.details.get("seed_landed")
            if isinstance(landed, list):
                rows.append(
                    _detail_row("Landings", f"{sum(1 for game in landed if game)}/{len(landed)}")
                )
        rows.append(_detail_row("Replay", html.escape(_replay_text(payload))))
    if result.benchmark in CHESS_BENCHMARKS:
        error = _error_text(result)
        rows.append(
            _detail_row("Mean error", html.escape(f"{error} pawns" if error != "-" else "-"))
        )
    tolerance = result.details.get("tolerance_pawns")
    if isinstance(tolerance, (int, float)):
        rows.append(_detail_row("Tolerance", f"{tolerance:g} pawns"))
    meta = "".join(rows)

    if payload is not None and result.benchmark in SNAKE_BENCHMARKS:
        size = int(payload.get("size", 8))
        canvas = _REPLAY_BOX
        default_speed = "16" if size == 16 else "4"
        script = (
            _GAME_SCRIPT.replace("__FRAMES__", json.dumps(payload["frames"]))
            .replace("__SIZE__", str(size))
            .replace("__BLOCKED__", json.dumps(payload.get("blocked", [])))
        )
        seed = int(payload.get("seed", 0))
        moves = int(payload.get("moves", 0))
        length = int(payload.get("length", 0))
        canvas_label = (
            f"Snake replay board. Seed {seed}, {moves:,} moves, final length {length}. "
            "Use Play and Step buttons below; current position is announced in status text."
        )
        canvas_fallback = f"Snake replay: Seed {seed}, {moves:,} moves, final length {length}. Enable JavaScript to play."
        game_section = (
            f'<section class="detail-section detail-game" aria-labelledby="replay-title">'
            f'<div class="detail-section-head"><h2 id="replay-title">Replay</h2>'
            f"<p>Seed {seed} · {moves:,} moves · final length {length}</p></div>"
            f'<canvas id="board" width="{canvas}" height="{canvas}" role="img" '
            f'aria-label="{html.escape(canvas_label)}" aria-describedby="status">{html.escape(canvas_fallback)}</canvas>'
            '<div class="controls"><button id="play" type="button">Play</button>'
            '<button id="prev" type="button">Step back</button>'
            '<button id="next" type="button">Step forward</button>'
            f"{_speed_select(default_speed)}</div>"
            '<noscript><p class="muted">Replay needs JavaScript. Summary above gives seed, moves and final length.</p></noscript>'
            '<p id="status" class="muted" aria-live="polite"></p></section>'
        )
        script_tag = f"<script>{script}</script>"
        grid_class = "detail-grid"
    elif payload is not None and result.benchmark in RL_BENCHMARKS:
        game_section, control_script = _control_game_section(result.benchmark, payload)
        script_tag = f"<script>{control_script}</script>"
        grid_class = "detail-grid"
    else:
        game_section = ""
        script_tag = ""
        grid_class = "detail-grid detail-grid--single"

    eyebrow_html = f'<span class="detail-group">{html.escape(group_label)}</span> · {html.escape(version_label)}'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} — {html.escape(version_label)} — Minimal NN Competition</title><style>{_STYLE}{_DETAIL_STYLE}</style></head><body data-accent="{html.escape(accent)}"><div class="detail-shell"><header class="detail-head"><p class="eyebrow">{eyebrow_html}</p><h1 class="detail-title">{html.escape(title)}</h1><p class="detail-subtitle">Submission details</p><p class="detail-links"><a href="../">← Back to leaderboard</a><span>·</span><a href="../leaderboard.json">Raw leaderboard JSON</a></p></header><div class="{grid_class}">{game_section}<aside class="detail-panel" aria-labelledby="details-title"><h2 class="detail-kicker" id="details-title">Submission</h2><dl class="detail-meta">{meta}</dl></aside></div><footer class="detail-footer"><a href="../">Back to leaderboard</a><span>Minimal NN Competition · <a href="../internal/speed/" title="Hidden Snake Simple 8x8 environment speed leaderboard">speed board</a></span></footer></div>{script_tag}</body></html>"""


def render_submission_pages(results, replays, output_dir):
    """Write one detail page per submission; return benchmark/submission links.

    Replay JSON is written under ``games/`` for Snake and control replays.
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
        payload = replays.get(stem)
        if payload is not None and result.benchmark in SNAKE_BENCHMARKS + RL_BENCHMARKS:
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
    group_stats = {}
    for spec in groups:
        group_entries = [
            result
            for benchmark in spec["benchmarks"]
            for result in entries.get(benchmark, ())
        ]
        group_stats[spec["id"]] = (
            len(group_entries),
            sum(1 for result in group_entries if result.passed),
        )
    group_pass_percent = {
        group_id: _pass_percent(group_passed, group_total)
        for group_id, (group_total, group_passed) in group_stats.items()
    }
    brand_dots = "".join(
        f'<span data-accent="{html.escape(spec["accent"], quote=True)}" '
        f'style="--pass-percent:{group_pass_percent[spec["id"]]}"></span>'
        for spec in groups
    )
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
        group_count, group_passed = group_stats[group_id]
        pass_percent = _pass_percent(group_passed, group_count)
        rail_links.append(
            f'<a href="#{html.escape(group_id)}" data-accent="{html.escape(accent)}">'
            f'<span class="rail__dot" style="--pass-percent:{pass_percent}" aria-hidden="true"></span>'
            f'<span class="rail__index">{group_number:02d}</span><strong>{html.escape(label)}</strong>'
            f'<small>{group_passed} / {group_count} passing</small></a>'
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
                f'<p class="sr-only" aria-live="polite"><span data-current>'
                f"{html.escape(default_label)} · 1 of {len(versions)}</span></p>"
            )
        else:
            version_controls = ""
            hint = ""
        version_row = (
            f'<div class="version-row">{version_controls}{hint}</div>'
            if (version_controls or hint)
            else ""
        )
        panels = []
        for benchmark in versions:
            version_entries = entries.get(benchmark, [])
            version_label = _version_label(benchmark)
            active = benchmark == default_benchmark
            table = _table(version_entries, submission_links, benchmark)
            panels.append(
                f'<div class="version-panel{" is-active" if active else ""}" id="panel-{html.escape(benchmark)}" data-version="{html.escape(benchmark)}" data-label="{html.escape(version_label)}">'
                f'<label class="passed-toggle"><input type="checkbox" data-passed-toggle checked> Pass only</label>'
                f'<div class="table-wrap" role="region" tabindex="0" aria-label="{html.escape(version_label)} results">{table}</div>'
                f"</div>"
            )
        sections.append(
            f'<section class="group" id="{html.escape(group_id)}" data-group-section data-accent="{html.escape(accent)}" aria-labelledby="{html.escape(group_id)}-title">'
            f'<div class="group__head"><div><p class="eyebrow" id="{html.escape(group_id)}-title"><span class="group__index">{group_number:02d}</span> / {html.escape(label)}</p></div></div>'
            f"{version_row}{''.join(panels)}</section>"
        )
    rail_note = f"{total} {_plural(total, 'submission')} total · {passed} pass" if groups else ""
    content = "".join(sections) or "<p>No results have been recorded yet.</p>"
    featured = min(
        (
            result
            for result in results
            if result.benchmark == "mnist" and result.passed
        ),
        key=lambda result: result.parameter_count,
        default=None,
    )
    featured_html = ""
    if featured:
        featured_href = (submission_links or {}).get(
            (featured.benchmark, featured.submission)
        )
        featured_name = html.escape(featured.name)
        if featured_href:
            featured_name = (
                f'<a href="{html.escape(featured_href, quote=True)}">'
                f"{featured_name}</a>"
            )
        view_featured = (
            f'<a class="featured__link" href="{html.escape(featured_href, quote=True)}">'
            'View submission <span aria-hidden="true">→</span></a>'
            if featured_href
            else ""
        )
        featured_html = (
            '<section class="featured" aria-labelledby="featured-title">'
            '<div class="featured__intro">'
            '<p class="eyebrow">Model spotlight · smallest passing on MNIST</p>'
            f'<h2 id="featured-title">{featured_name}</h2></div>'
            '<div class="featured__stats" aria-label="MNIST result">'
            f'<p class="featured__stat"><strong>{html.escape(_score_text(featured))}</strong>'
            '<span>MNIST score</span></p>'
            f'<p class="featured__stat"><strong>{featured.parameter_count:,}</strong>'
            '<span>parameters</span></p></div>'
            f"{view_featured}</section>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Minimal NN Competition</title><script>document.documentElement.classList.add("js");</script><style>{_STYLE}{_SITE_STYLE}</style></head><body><div class="shell"><aside class="rail"><p class="brand-mark"><span class="brand-dots" aria-hidden="true">{brand_dots}</span>Minimal NN Competition</p><p class="rail__label">Benchmark groups</p><nav aria-label="Benchmark groups">{"".join(rail_links)}</nav><p class="rail__note">{rail_note}</p></aside><main class="main"><header class="masthead"><p class="eyebrow">Open challenge · Public leaderboard</p><h1>How small can a model get?</h1><p class="lead">Build a compact network, test it against the benchmarks, and see where it ranks.</p><p class="links"><a class="primary-action" href="https://github.com/wd7512/minimal-nn-competition/blob/main/CONTRIBUTING.md">Submit your model <span aria-hidden="true">→</span></a></p></header>{featured_html}{content}<footer class="footer"><p>Minimal NN Competition · static results</p><a class="version" href="internal/speed/" title="Hidden Snake Simple 8x8 environment speed leaderboard" aria-label="Hidden Snake Simple 8x8 environment speed leaderboard">v{html.escape(__version__)}</a></footer></main></div><script>{_SITE_SCRIPT}</script></body></html>
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
