"""Measure Snake environment throughput with a fixed deterministic agent."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.speed.agent import ACTIONS_BY_CELL, AGENT_NAME, BOARD_SIZE, SIZE

HARNESS_VERSION = "snake-speed-v1"
DEFAULT_DURATION = 1.0
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 1.0
TIMER_CHUNK = 1024
VALIDATION_SEEDS = tuple(range(100))


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            if completed.stdout.strip():
                return completed.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    try:
        lines = cpuinfo.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _run_once(seconds: float) -> tuple[int, float]:
    environment = SnakeEnvironment(size=SIZE, seed=0, max_moves=4096)
    observation = environment.reset()
    step = environment.step
    reset = environment.reset
    actions = ACTIONS_BY_CELL
    start = time.perf_counter()
    deadline = start + seconds
    ticks = 0
    while True:
        for _ in range(TIMER_CHUNK):
            observation, _, done = step(actions[observation[1] * SIZE + observation[0]])
            ticks += 1
            if done:
                observation = reset()
        if time.perf_counter() >= deadline:
            return ticks, time.perf_counter() - start


def _validate_agent() -> None:
    for seed in VALIDATION_SEEDS:
        environment = SnakeEnvironment(size=SIZE, seed=seed, max_moves=4096)
        observation = environment.reset()
        done = False
        while not done:
            observation, _, done = environment.step(
                ACTIONS_BY_CELL[observation[1] * SIZE + observation[0]]
            )
        if len(environment.snake) != BOARD_SIZE:
            raise RuntimeError(f"Speed agent failed to fill the board on seed {seed}")


def measure_snake_speed(
    duration: float = DEFAULT_DURATION,
    runs: int = DEFAULT_RUNS,
    warmup: float = DEFAULT_WARMUP,
    validate: bool = True,
) -> SpeedResult:
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
