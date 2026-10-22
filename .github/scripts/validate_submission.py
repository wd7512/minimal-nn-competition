#!/usr/bin/env python3
"""Validate that a pull request changes only one submission directory."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SUBMISSION_PATH = re.compile(r"^models/([^/]+)/([^/]+)/(config\.json|model\.pt)$")
ALLOWED_STATUSES = {"A", "M", "C"}
ALLOWED_ARCHITECTURES = {"linear", "mlp"}
ALLOWED_ACTIVATIONS = {"relu", "identity"}


def _validate_config(config: Any, directory: Path) -> list[str]:
    config_path = directory / "config.json"
    if not isinstance(config, dict):
        return [f"{config_path}: must contain a JSON object"]

    errors: list[str] = []
    benchmark = directory.parts[1] if len(directory.parts) > 1 else ""
    if config.get("benchmark") != benchmark:
        errors.append(f"{config_path}: benchmark must be {benchmark!r}")
    for field in ("name", "author"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{config_path}: {field} must be a non-empty string")
    if config.get("architecture") not in ALLOWED_ARCHITECTURES:
        errors.append(f"{config_path}: architecture must be linear or mlp")
    input_size = config.get("input_size")
    if type(input_size) is not int or input_size <= 0:
        errors.append(f"{config_path}: input_size must be a positive integer")
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
    if type(config.get("normalisation", False)) is not bool:
        errors.append(f"{config_path}: normalisation must be a boolean")
    return errors


def validate_diff(diff_text: str, root: Path) -> list[str]:
    """Return validation errors for a ``git diff --name-status`` payload."""
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
            continue
        if SUBMISSION_PATH.match(path) is None:
            errors.append(
                f"{path}: only models/<benchmark>/<name>/config.json and model.pt are allowed"
            )
            continue
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
            errors.extend(_validate_config(config, directory))
    if not model_path.is_file():
        errors.append(f"{model_path}: missing")
    elif model_path.stat().st_size == 0:
        errors.append(f"{model_path}: must not be empty")
    return errors


def main() -> int:
    base_sha = os.environ.get("BASE_SHA")
    if not base_sha:
        sys.stderr.write("BASE_SHA is required\n")
        return 2
    diff = subprocess.run(
        ["git", "diff", "--name-status", f"{base_sha}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    errors = validate_diff(diff, Path.cwd())
    for error in errors:
        sys.stdout.write(f"::error::{error}\n")
    if errors:
        return 1
    sys.stdout.write("Submission check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
