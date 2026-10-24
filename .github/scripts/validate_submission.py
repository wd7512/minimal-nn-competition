#!/usr/bin/env python3
"""Validate that a pull request changes only one submission directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from minimal_nn_competition.submissions.validation import validate_diff


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
