"""Validation shared by local acceptance and GitHub submission checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
from minimal_nn_competition.benchmarks.snake.observation import OBSERVATION_SIZE
from minimal_nn_competition.model.entry import SubmissionConfig
from minimal_nn_competition.model.limits import (
    MAX_CHECKPOINT_BYTES,
    MAX_PARAMETERS,
    parameter_count_from_dimensions,
)

SUBMISSION_PATH = re.compile(r"^models/([^/]+)/([^/]+)/(config\.json|model\.pt)$")
ALLOWED_STATUSES = {"A", "M", "C"}
ALLOWED_ARCHITECTURES = {"linear", "mlp"}
ALLOWED_ACTIVATIONS = {"relu", "identity"}
ALLOWED_CONFIG_KEYS = set(SubmissionConfig.model_fields)


def _validate_input_indices(
    config: dict[str, Any],
    config_path: Path,
    input_size: Any,
    is_snake: bool = False,
) -> list[str]:
    indices = config.get("input_indices")
    if indices is None:
        return []
    if indices is False:
        return [f"{config_path}: input_indices must be true or a list of feature indices"]
    if indices is True:
        if not is_snake:
            return [f"{config_path}: input_indices is only supported for snake benchmarks"]
        return []
    if not isinstance(indices, list) or not indices:
        return [f"{config_path}: input_indices must be a non-empty list of feature indices"]
    if any(type(index) is not int or index < 0 for index in indices):
        return [f"{config_path}: input_indices must contain non-negative integers"]
    if len(set(indices)) != len(indices):
        return [f"{config_path}: input_indices must not contain duplicates"]
    if type(input_size) is int and input_size > 0 and len(indices) != input_size:
        return [f"{config_path}: input_indices length must match input_size"]
    if is_snake and any(index >= OBSERVATION_SIZE for index in indices):
        return [f"{config_path}: input_indices must be in 0..{OBSERVATION_SIZE - 1}"]
    if not is_snake:
        return [f"{config_path}: input_indices is only supported for snake benchmarks"]
    return []


def validate_config(config: Any, directory: Path) -> list[str]:
    config_path = directory / "config.json"
    if not isinstance(config, dict):
        return [f"{config_path}: must contain a JSON object"]
    errors: list[str] = []
    benchmark = directory.parts[1] if len(directory.parts) > 1 else ""
    if config.get("benchmark") != benchmark:
        errors.append(f"{config_path}: benchmark must be {benchmark!r}")
    unknown_keys = sorted(set(config) - ALLOWED_CONFIG_KEYS)
    if unknown_keys:
        errors.append(f"{config_path}: unknown config keys: {', '.join(unknown_keys)}")
    for field in ("name", "author"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{config_path}: {field} must be a non-empty string")
    if not isinstance(config.get("description", ""), str):
        errors.append(f"{config_path}: description must be a string")
    if config.get("architecture") not in ALLOWED_ARCHITECTURES:
        errors.append(f"{config_path}: architecture must be linear or mlp")
    input_size = config.get("input_size")
    if type(input_size) is not int or input_size <= 0:
        errors.append(f"{config_path}: input_size must be a positive integer")
    errors.extend(
        _validate_input_indices(
            config,
            config_path,
            input_size,
            is_snake=config.get("benchmark") in SNAKE_BENCHMARKS,
        )
    )
    output_size = config.get("output_size")
    if type(output_size) is not int or output_size <= 0:
        errors.append(f"{config_path}: output_size must be a positive integer")
    hidden_sizes = config.get("hidden_sizes", [])
    if not isinstance(hidden_sizes, list) or any(
        type(size) is not int or size <= 0 for size in hidden_sizes
    ):
        errors.append(f"{config_path}: hidden_sizes must be positive integers")
    elif config.get("architecture") == "linear" and hidden_sizes:
        errors.append(f"{config_path}: linear submissions cannot declare hidden_sizes")
    if config.get("activation", "relu") not in ALLOWED_ACTIVATIONS:
        errors.append(f"{config_path}: activation must be relu or identity")

    # Estimate parameters from the raw config before building anything.
    input_size = config.get("input_size")
    output_size = config.get("output_size")
    hidden_sizes = config.get("hidden_sizes", [])
    if (
        type(input_size) is int
        and input_size > 0
        and type(output_size) is int
        and output_size > 0
        and isinstance(hidden_sizes, list)
        and all(type(size) is int and size > 0 for size in hidden_sizes)
        and config.get("architecture") in {"linear", "mlp"}
    ):
        total = parameter_count_from_dimensions(input_size, hidden_sizes, output_size)
        if total > MAX_PARAMETERS:
            errors.append(
                f"{config_path}: submission would require {total} parameters, "
                f"above the {MAX_PARAMETERS} maximum"
            )
    if type(config.get("normalisation", False)) is not bool:
        errors.append(f"{config_path}: normalisation must be a boolean")
    return errors


def validate_diff(diff_text: str, root: Path) -> list[str]:
    """Validate a ``git diff --name-status`` payload."""
    errors: list[str] = []
    paths: list[str] = []
    for line in diff_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            errors.append(f"malformed diff line: {line!r}")
            continue
        status, path = parts[0], parts[-1]
        if status[:1] not in ALLOWED_STATUSES:
            errors.append(f"{path}: only added or modified files are allowed")
        elif SUBMISSION_PATH.match(path) is None:
            errors.append(
                f"{path}: only models/<benchmark>/<name>/config.json and model.pt are allowed"
            )
        else:
            paths.append(path)
    if len(paths) != 2:
        errors.append(f"expected exactly two submission files, found {len(paths)}")
        return errors
    if {Path(path).name for path in paths} != {"config.json", "model.pt"}:
        errors.append("a submission must contain exactly config.json and model.pt")
        return errors
    directories = {Path(path).parent for path in paths}
    if len(directories) != 1:
        errors.append("config.json and model.pt must be in the same submission directory")
        return errors
    directory = directories.pop()
    config_path = root / directory / "config.json"
    model_path = root / directory / "model.pt"
    if not config_path.is_file():
        errors.append(f"{config_path}: missing")
    else:
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{config_path}: invalid JSON ({exc})")
        else:
            errors.extend(validate_config(config, directory))
    if not model_path.is_file():
        errors.append(f"{model_path}: missing")
    elif model_path.stat().st_size == 0:
        errors.append(f"{model_path}: must not be empty")
    elif model_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        errors.append(
            f"{model_path}: checkpoint is too large "
            f"({model_path.stat().st_size} bytes; maximum is {MAX_CHECKPOINT_BYTES})"
        )
    return errors


def submission_directory(diff_text: str) -> Path:
    paths = [line.split("\t")[-1] for line in diff_text.splitlines() if "\t" in line]
    directories = {Path(path).parent for path in paths if SUBMISSION_PATH.match(path)}
    if len(directories) != 1:
        raise ValueError("diff does not identify one submission directory")
    return directories.pop()
