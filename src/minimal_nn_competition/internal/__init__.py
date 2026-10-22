"""Deterministic token-count leaderboards for the repo's own code (issue #6)."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from minimal_nn_competition.internal.scoring import RULES_VERSION, score_files

logger = logging.getLogger(__name__)

# Set in validation subprocesses so tests that invoke `mnc internal` itself
# skip instead of recursing (the outer run already covers them).
_NESTED_ENV_VAR = "MINIMAL_NN_COMPETITION_INTERNAL"

PACKAGE = "src/minimal_nn_competition"
_MNIST_FILES = (f"{PACKAGE}/benchmarks/mnist", f"{PACKAGE}/examples/mnist_submission.py")
_SNAKE_FILES = (f"{PACKAGE}/benchmarks/snake", f"{PACKAGE}/examples/snake_submission.py")


@dataclass(frozen=True)
class Scope:
    """One internal leaderboard: source prefixes plus pytest node IDs."""

    name: str
    sources: tuple[str, ...]
    tests: tuple[str, ...]


SCOPES: tuple[Scope, ...] = (
    Scope(
        name="mnist",
        sources=_MNIST_FILES,
        tests=(
            "tests/test_benchmarks.py::test_mnist_evaluation_and_threshold",
            "tests/test_benchmarks.py::test_mnist_normalisation_flag",
            "tests/test_benchmarks.py::test_mnist_requires_data_file",
            "tests/test_benchmarks.py::test_mnist_rejects_wrong_shape",
            "tests/test_mnist_data.py",
        ),
    ),
    Scope(
        name="snake",
        sources=_SNAKE_FILES,
        tests=(
            "tests/test_benchmarks.py::test_snake_partial_run_does_not_pass",
            "tests/test_benchmarks.py::test_snake_full_board_passes",
            "tests/test_benchmarks.py::test_snake_defaults_to_100_seeds",
            "tests/test_benchmarks.py::test_snake_score_std_across_seeds",
            "tests/test_benchmarks.py::test_snake_rejects_wrong_config",
            "tests/test_benchmarks.py::test_snake_selected_input_indices_build_smaller_model",
            "tests/test_benchmarks.py::test_snake_selected_inputs_evaluate",
            "tests/test_benchmarks.py::test_snake_rejects_out_of_range_input_indices",
            "tests/test_core.py::test_snake_is_deterministic",
            "tests/test_core.py::test_snake_can_move_into_vacating_tail",
            "tests/test_integration.py::test_evaluate_snake_cli",
            "tests/test_integration.py::test_data_path_rejected_for_snake",
            "tests/test_observation.py",
            "tests/test_snake_replay.py",
        ),
    ),
    Scope(
        name="core",
        sources=(PACKAGE,),
        tests=(
            "tests/test_core.py::test_parameter_count_rejects_non_tensors",
            "tests/test_core.py::test_parameter_count_sums_tensor_values",
            "tests/test_core.py::test_config_round_trip",
            "tests/test_core.py::test_config_accepts_input_indices",
            "tests/test_core.py::test_config_rejects_duplicate_input_indices",
            "tests/test_core.py::test_config_rejects_linear_hidden_sizes",
            "tests/test_core.py::test_registry_and_leaderboard_ranking",
            "tests/test_core.py::test_leaderboard_round_trip",
            "tests/test_core.py::test_leaderboard_cli",
            "tests/test_import.py",
            "tests/test_integration.py::test_checked_in_examples_load",
            "tests/test_integration.py::test_render_site",
            "tests/test_integration.py::test_render_cli",
            "tests/test_integration.py::test_collect_cli",
            "tests/test_internal.py",
            "tests/test_model.py",
            "tests/test_speed.py",
            "tests/test_submission_check.py",
        ),
    ),
)


def expand_scope_tests(scope: Scope, collected: Iterable[str]) -> list[str]:
    """Expand a scope's entries (files and node IDs) against collected IDs."""
    nodes = []
    for entry in scope.tests:
        if "::" in entry:
            nodes.append(entry)
        else:
            nodes.extend(sorted(node for node in collected if node.startswith(entry + "::")))
    return nodes


def scope_files(scope: Scope, root: Path) -> list[Path]:
    """Resolve a scope's source prefixes to Python files under ``root``."""
    excluded = [root / source for source in _MNIST_FILES + _SNAKE_FILES]
    files = []
    for source in scope.sources:
        path = root / source
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for candidate in candidates:
            if candidate.suffix != ".py" or "__pycache__" in candidate.parts:
                continue
            if scope.name == "core" and any(
                candidate == excluded_path or excluded_path in candidate.parents
                for excluded_path in excluded
            ):
                continue
            files.append(candidate)
    if not files:
        raise ValueError(f"Scope {scope.name!r} matched no Python files")
    return sorted(files)


def check_interpreter(root: Path) -> str:
    """Require the running interpreter to match the pinned ``.python-version``."""
    pinned = (root / ".python-version").read_text(encoding="utf-8").strip()
    running = platform.python_version()
    if not running.split(".")[:2] == pinned.split(".")[:2]:
        raise RuntimeError(
            f"Scoring requires Python {pinned} (see .python-version), but running {running}"
        )
    return running


def run_scope_tests(scope: Scope, root: Path, timeout: int = 600) -> tuple[bool, str]:
    """Run a scope's tests; return (passed, failure reason)."""
    env = {**os.environ, _NESTED_ENV_VAR: "1"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *scope.tests],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run tests: {exc}"
    if completed.returncode == 0:
        return True, ""
    tail = "\n".join(completed.stdout.splitlines()[-5:]).strip()
    return False, tail or f"pytest exited with status {completed.returncode}"


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _git_dirty(root: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(completed.stdout.strip())


@dataclass(frozen=True)
class InternalResult:
    scope: str
    tokens: int
    commit: str
    dirty: bool
    measured_at: str
    machine: str
    python_version: str
    rules_version: int = RULES_VERSION
    status: str = "pass"
    failure_reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def measure_scope(scope: Scope, root: Path) -> InternalResult:
    """Validate a scope's tests, then score its sources."""
    python_version = check_interpreter(root)
    passed, reason = run_scope_tests(scope, root)
    tokens = score_files(scope_files(scope, root)) if passed else 0
    if not passed:
        logger.warning("Scope %s failed validation: %s", scope.name, reason)
    return InternalResult(
        scope=scope.name,
        tokens=tokens,
        commit=_git_commit(root),
        dirty=_git_dirty(root),
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        machine=platform.machine() or "unknown",
        python_version=python_version,
        status="pass" if passed else "fail",
        failure_reason="" if passed else reason,
    )


def rank_internal_results(results: Iterable[InternalResult]) -> list[InternalResult]:
    """Rank passing entries per scope by ascending tokens; failures last."""
    return sorted(
        results,
        key=lambda result: (
            result.scope,
            result.status != "pass",
            result.tokens,
            result.commit,
        ),
    )


def upsert_internal_result(
    results: Iterable[InternalResult], result: InternalResult
) -> list[InternalResult]:
    """Replace the entry for (scope, commit), keeping everything else."""
    kept = [
        existing
        for existing in results
        if (existing.scope, existing.commit) != (result.scope, result.commit)
    ]
    return kept + [result]


def save_internal_results(path: Path, results: Iterable[InternalResult]) -> None:
    payload = [asdict(result) for result in rank_internal_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_internal_results(path: Path) -> list[InternalResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Internal leaderboard must contain a JSON list")
    results = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Internal leaderboard entries must be JSON objects")
        results.append(
            InternalResult(
                scope=str(item["scope"]),
                tokens=int(item["tokens"]),
                commit=str(item.get("commit", "unknown")),
                dirty=bool(item.get("dirty", False)),
                measured_at=str(item.get("measured_at", "")),
                machine=str(item.get("machine", "")),
                python_version=str(item.get("python_version", "")),
                rules_version=int(item.get("rules_version", RULES_VERSION)),
                status=str(item.get("status", "pass")),
                failure_reason=str(item.get("failure_reason", "")),
            )
        )
    return rank_internal_results(results)


def scope_rank(result: InternalResult, results: Iterable[InternalResult]) -> int:
    """Competition rank within a scope: 1 + number of strictly lower scores."""
    peers = [
        other.tokens for other in results if other.scope == result.scope and other.status == "pass"
    ]
    return 1 + sum(1 for tokens in peers if tokens < result.tokens)
