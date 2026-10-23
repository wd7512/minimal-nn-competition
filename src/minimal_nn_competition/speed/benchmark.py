"""Measure Snake Simple 8x8 environment throughput with a fixed deterministic agent."""

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from minimal_nn_competition.benchmarks.snake.engine import SnakeEngine
from minimal_nn_competition.speed.agent import ACTIONS_BY_CELL, AGENT_NAME, BOARD_SIZE, SIZE

HARNESS_VERSION = "snake-simple-8x8-speed-v2"
DEFAULT_DURATION = 1.0
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 1.0
TIMER_CHUNK = 1024
VALIDATION_SEEDS = range(100)


@dataclass
class SpeedResult:
    commit: str
    ticks_per_second: float
    ticks_std: float
    run_ticks: list[int]
    measured_at: str
    duration_seconds: float
    runs: int
    warmup_seconds: float
    agent: str
    harness: str
    python_version: str
    platform: str
    machine: str
    cpu: str
    validation_seeds: int

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_name():
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            if cpu:
                return cpu
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _run_once(seconds):
    engine = SnakeEngine(size=SIZE, seed=0, max_moves=4096)
    engine.reset()
    step = engine.step
    reset = engine.reset
    actions = ACTIONS_BY_CELL
    start = time.perf_counter()
    deadline = start + seconds
    ticks = 0
    while True:
        for _ in range(TIMER_CHUNK):
            _, done = step(actions[engine.head_cell])
            ticks += 1
            if done:
                reset()
        if time.perf_counter() >= deadline:
            return ticks, time.perf_counter() - start


def _validate_agent():
    for seed in VALIDATION_SEEDS:
        engine = SnakeEngine(size=SIZE, seed=seed, max_moves=4096)
        done = False
        while not done:
            _, done = engine.step(ACTIONS_BY_CELL[engine.head_cell])
        if len(engine.snake) != BOARD_SIZE:
            raise RuntimeError(f"Speed agent failed to fill the board on seed {seed}")


def measure_snake_speed(
    duration=DEFAULT_DURATION,
    runs=DEFAULT_RUNS,
    warmup=DEFAULT_WARMUP,
    validate=True,
):
    if not duration > 0:
        raise ValueError("Speed benchmark duration must be positive")
    if runs <= 0:
        raise ValueError("Speed benchmark runs must be positive")
    if not warmup >= 0:
        raise ValueError("Speed benchmark warmup must be non-negative")
    if validate:
        _validate_agent()
    if warmup > 0:
        _run_once(warmup)
    run_ticks = []
    rates = []
    for _ in range(runs):
        ticks, elapsed = _run_once(duration)
        run_ticks.append(ticks)
        rates.append(ticks / elapsed)
    return SpeedResult(
        commit=_git_commit(),
        ticks_per_second=statistics.fmean(rates),
        ticks_std=statistics.pstdev(rates),
        run_ticks=run_ticks,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_seconds=duration,
        runs=runs,
        warmup_seconds=warmup,
        agent=AGENT_NAME,
        harness=HARNESS_VERSION,
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=_cpu_name(),
        validation_seeds=len(VALIDATION_SEEDS) if validate else 0,
    )
