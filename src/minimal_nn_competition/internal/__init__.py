"""Deterministic token-count leaderboards for the repo's own code (issue #6)."""

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from minimal_nn_competition.internal.scoring import RULES_VERSION, score_files

logger = logging.getLogger(__name__)

# Set in validation subprocesses so tests that invoke `mnnc internal` itself
# skip instead of recursing (the outer run already covers them).
_NESTED_ENV_VAR = "MINIMAL_NN_COMPETITION_INTERNAL"

PACKAGE = "src/minimal_nn_competition"
_SHARED_IMAGE_FILES = (
    f"{PACKAGE}/benchmarks/idx.py",
    f"{PACKAGE}/benchmarks/image_classification.py",
    f"{PACKAGE}/examples/classifier_submission.py",
)
_MNIST_FILES = (
    f"{PACKAGE}/benchmarks/mnist",
    *_SHARED_IMAGE_FILES,
    f"{PACKAGE}/examples/mnist_submission.py",
)
_FASHION_FILES = (
    f"{PACKAGE}/benchmarks/fashion_mnist",
    *_SHARED_IMAGE_FILES,
    f"{PACKAGE}/examples/fashion_mnist_submission.py",
)
_SNAKE_FILES = (f"{PACKAGE}/benchmarks/snake", f"{PACKAGE}/examples/snake_submission.py")
_CHESS_FILES = (
    f"{PACKAGE}/benchmarks/chess_eval",
    f"{PACKAGE}/examples/chess_eval_submission.py",
)
_CARTPOLE_FILES = (
    f"{PACKAGE}/benchmarks/cartpole",
    f"{PACKAGE}/benchmarks/control",
    f"{PACKAGE}/examples/cartpole_submission.py",
)
_PENDULUM_FILES = (
    f"{PACKAGE}/benchmarks/pendulum",
    f"{PACKAGE}/examples/pendulum_submission.py",
    f"{PACKAGE}/examples/ppo.py",
)
_LUNARLANDER_FILES = (
    f"{PACKAGE}/benchmarks/lunarlander",
    f"{PACKAGE}/examples/lunarlander_submission.py",
    f"{PACKAGE}/examples/ppo.py",
)


@dataclass(frozen=True)
class Scope:
    """One internal leaderboard: source prefixes plus pytest node IDs."""

    name: str
    sources: tuple[str, ...]
    tests: tuple[str, ...]


SCOPES = (
    Scope(
        "mnist",
        _MNIST_FILES,
        tuple(
            """tests/test_benchmarks.py::test_mnist_evaluation_and_threshold
tests/test_benchmarks.py::test_mnist_normalisation_flag
tests/test_benchmarks.py::test_mnist_requires_data_file
tests/test_benchmarks.py::test_mnist_rejects_wrong_shape
tests/test_benchmarks.py::test_mnist_accepts_two_dimensional_labels
tests/test_benchmarks.py::test_mnist_rejects_length_mismatch
tests/test_benchmarks.py::test_mnist_rejects_wrong_input_width
tests/test_benchmarks.py::test_mnist_rejects_partial_dataset
tests/test_benchmarks.py::test_mnist_rejects_legacy_test_cache
tests/test_benchmarks.py::test_image_benchmarks_expect_full_dataset
tests/test_mnist_data.py
tests/test_integration.py::test_evaluate_mnist_cli
tests/test_integration.py::test_evaluate_mnist_uses_default_data_path
tests/test_integration.py::test_collect_mnist_cli
tests/test_integration.py::test_collect_rejects_partial_mnist_dataset""".splitlines()
        ),
    ),
    Scope(
        "fashion-mnist",
        _FASHION_FILES,
        tuple(
            """tests/test_benchmarks.py::test_fashion_mnist_evaluation_and_threshold
tests/test_benchmarks.py::test_fashion_mnist_normalisation_flag
tests/test_benchmarks.py::test_fashion_mnist_accepts_two_dimensional_labels
tests/test_benchmarks.py::test_fashion_mnist_rejects_length_mismatch
tests/test_benchmarks.py::test_fashion_mnist_rejects_wrong_input_width
tests/test_benchmarks.py::test_fashion_mnist_rejects_partial_dataset
tests/test_benchmarks.py::test_fashion_mnist_rejects_legacy_test_cache
tests/test_benchmarks.py::test_fashion_mnist_requires_data_file
tests/test_benchmarks.py::test_fashion_mnist_rejects_wrong_shape
tests/test_fashion_mnist_data.py
tests/test_integration.py::test_collect_fashion_mnist_cli
tests/test_integration.py::test_evaluate_fashion_mnist_cli
tests/test_integration.py::test_evaluate_fashion_mnist_uses_default_data_path
tests/test_integration.py::test_collect_rejects_partial_fashion_mnist_dataset
tests/test_training.py""".splitlines()
        ),
    ),
    Scope(
        "snake-simple-8x8",
        _SNAKE_FILES,
        tuple(
            """tests/test_benchmarks.py::test_snake_partial_run_does_not_pass
tests/test_benchmarks.py::test_snake_full_board_passes
tests/test_benchmarks.py::test_snake_defaults_to_100_seeds
tests/test_benchmarks.py::test_snake_score_std_across_seeds
tests/test_benchmarks.py::test_snake_rejects_wrong_config
tests/test_benchmarks.py::test_snake_selected_input_indices_build_smaller_model
tests/test_benchmarks.py::test_snake_selected_inputs_evaluate
tests/test_benchmarks.py::test_snake_rejects_out_of_range_input_indices
tests/test_core.py::test_snake_is_deterministic
tests/test_core.py::test_snake_can_move_into_vacating_tail
tests/test_integration.py::test_evaluate_snake_cli
tests/test_integration.py::test_data_path_rejected_for_snake
tests/test_observation.py
tests/test_snake_16x16.py
tests/test_snake_replay.py
tests/test_snake_obstacles.py""".splitlines()
        ),
    ),
    Scope(
        "chess-eval",
        _CHESS_FILES,
        tuple(
            """tests/test_benchmarks.py::test_chess_eval_registers_by_name
tests/test_benchmarks.py::test_chess_eval_encodes_fen
tests/test_benchmarks.py::test_chess_eval_load_tensors
tests/test_benchmarks.py::test_chess_eval_evaluate_dummy_passes
tests/test_benchmarks.py::test_chess_eval_evaluate_dummy_fails
tests/test_benchmarks.py::test_chess_eval_rejects_wrong_shape
tests/test_benchmarks.py::test_chess_eval_rejects_wrong_benchmark
tests/test_benchmarks.py::test_chess_eval_requires_data_file
tests/test_integration.py::test_evaluate_chess_eval_cli
tests/test_chess_eval_data.py::test_chess_eval_dataset_integrity
tests/test_chess_eval_data.py::test_chess_eval_100k_contains_10k""".splitlines()
        ),
    ),
    Scope(
        "cartpole",
        _CARTPOLE_FILES,
        tuple(
            """tests/test_rl_environments.py::test_cartpole_engine_is_deterministic
tests/test_rl_environments.py::test_cartpole_reset_bounds
tests/test_rl_environments.py::test_registry_includes_control_benchmarks
tests/test_rl_environments.py::test_cartpole_rejects_wrong_config
tests/test_rl_environments.py::test_cartpole_calibration
tests/test_rl_environments.py::test_control_evaluate_cli
tests/test_rl_environments.py::test_control_collect_cli
tests/test_rl_environments.py::test_control_train_commands_wire_examples
tests/test_rl_environments.py::test_render_control_group
tests/test_control_replay.py""".splitlines()
        ),
    ),
    Scope(
        "pendulum",
        _PENDULUM_FILES,
        tuple(
            """tests/test_rl_environments.py::test_pendulum_engine_is_deterministic
tests/test_rl_environments.py::test_pendulum_reset_bounds
tests/test_rl_environments.py::test_pendulum_rejects_wrong_config
tests/test_rl_environments.py::test_pendulum_calibration""".splitlines()
        ),
    ),
    Scope(
        "lunarlander",
        _LUNARLANDER_FILES,
        tuple(
            """tests/test_rl_environments.py::test_lunarlander_engine_is_deterministic
tests/test_rl_environments.py::test_lunarlander_reset_bounds
tests/test_rl_environments.py::test_lunarlander_rejects_wrong_config
tests/test_rl_environments.py::test_lunarlander_calibration""".splitlines()
        ),
    ),
    Scope(
        "core",
        (PACKAGE,),
        tuple(
            """tests/test_core.py::test_parameter_count_rejects_non_tensors
tests/test_core.py::test_parameter_count_sums_tensor_values
tests/test_core.py::test_config_round_trip
tests/test_core.py::test_config_description_is_optional
tests/test_core.py::test_config_accepts_input_indices
tests/test_core.py::test_config_rejects_duplicate_input_indices
tests/test_core.py::test_config_rejects_missing_identity
tests/test_core.py::test_config_rejects_linear_hidden_sizes
tests/test_core.py::test_registry_and_leaderboard_ranking
tests/test_core.py::test_leaderboard_round_trip
tests/test_core.py::test_leaderboard_cli
tests/test_import.py
tests/test_configs.py
tests/test_integration.py::test_checked_in_examples_load
tests/test_integration.py::test_render_site
tests/test_integration.py::test_render_near_perfect_failure_is_not_shown_as_100
tests/test_integration.py::test_render_mnist_has_no_game_length_column
tests/test_integration.py::test_render_chess_eval
tests/test_integration.py::test_rank_chess_eval_breaks_score_ties_by_mean_error
tests/test_integration.py::test_render_groups_versions_and_defaults
tests/test_integration.py::test_render_cli
tests/test_integration.py::test_render_shows_name_and_author
tests/test_integration.py::test_collect_cli
tests/test_internal.py
tests/test_model.py
tests/test_speed.py
tests/test_submission_check.py""".splitlines()
            + ["tests/test_submission_workflow.py"]
        ),
    ),
)


def expand_scope_tests(scope, collected):
    """Expand a scope's entries (files and node IDs) against collected IDs."""
    nodes = []
    for entry in scope.tests:
        nodes += (
            [entry]
            if "::" in entry
            else sorted(node for node in collected if node.startswith(f"{entry}::"))
        )
    return nodes


def scope_files(scope, root):
    """Resolve a scope's source prefixes to Python files under ``root``."""
    excluded = [
        root / source
        for source in _MNIST_FILES
        + _FASHION_FILES
        + _SNAKE_FILES
        + _CHESS_FILES
        + _CARTPOLE_FILES
        + _PENDULUM_FILES
        + _LUNARLANDER_FILES
    ]
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


def check_interpreter(root):
    """Require the running interpreter to match the pinned ``.python-version``."""
    pinned = (root / ".python-version").read_text(encoding="utf-8").strip()
    running = platform.python_version()
    if running.split(".")[:2] != pinned.split(".")[:2]:
        raise RuntimeError(
            f"Scoring requires Python {pinned} (see .python-version), but running {running}"
        )
    return running


def run_scope_tests(scope, root, timeout=600):
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


def _git(root, *args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, timeout=30, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None


def _git_commit(root):
    return _git(root, "rev-parse", "HEAD") or "unknown"


def _git_dirty(root):
    status = _git(root, "status", "--porcelain")
    return status is None or bool(status)


@dataclass(frozen=True)
class InternalResult:
    scope: str
    tokens: int
    commit: str = "unknown"
    dirty: bool = False
    measured_at: str = ""
    machine: str = ""
    python_version: str = ""
    rules_version: int = RULES_VERSION
    status: str = "pass"
    failure_reason: str = ""

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def measure_scope(scope, root):
    """Validate a scope's tests, then score its sources."""
    python_version = check_interpreter(root)
    passed, reason = run_scope_tests(scope, root)
    tokens = score_files(scope_files(scope, root)) if passed else 0
    if not passed:
        logger.warning("Scope %s failed validation: %s", scope.name, reason)
    return InternalResult(
        scope.name,
        tokens,
        _git_commit(root),
        _git_dirty(root),
        datetime.now(UTC).isoformat(timespec="seconds"),
        platform.machine() or "unknown",
        python_version,
        status="pass" if passed else "fail",
        failure_reason="" if passed else reason,
    )


def rank_internal_results(results):
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


def upsert_internal_result(results, result):
    """Replace the entry for (scope, commit), keeping everything else."""
    return [
        existing
        for existing in results
        if (existing.scope, existing.commit) != (result.scope, result.commit)
    ] + [result]


def save_internal_results(path, results):
    payload = [asdict(result) for result in rank_internal_results(results)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_internal_results(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Internal leaderboard must contain a JSON list")
    results = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Internal leaderboard entries must be JSON objects")
        results.append(InternalResult(**item))
    return rank_internal_results(results)


def scope_rank(result, results):
    """Competition rank within a scope: 1 + number of strictly lower scores."""
    return 1 + sum(
        other.tokens < result.tokens
        for other in results
        if other.scope == result.scope and other.status == "pass"
    )
