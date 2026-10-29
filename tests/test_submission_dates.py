"""Submission dates belong to each entry, including configs copied from older models."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimal_nn_competition.runner.entry import _submission_date


def _commit(root: Path, date: str) -> None:
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "Add model",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
    )


def test_submission_date_uses_copied_configs_own_addition(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    original = tmp_path / "old-model"
    original.mkdir()
    config = original / "config.json"
    config.write_text('{"benchmark": "mnist", "name": "model", "hidden_sizes": [8]}\n')
    _commit(tmp_path, "2026-10-22T12:00:00+00:00")

    copied = tmp_path / "new-model"
    copied.mkdir()
    (copied / "config.json").write_bytes(config.read_bytes())
    _commit(tmp_path, "2026-10-27T12:00:00+00:00")

    submission = SimpleNamespace(checkpoint=copied / "model.pt")
    assert _submission_date(submission) == "2026-10-27"

    (copied / "config.json").write_text('{"benchmark": "mnist", "name": "renamed model"}\n')
    _commit(tmp_path, "2026-10-28T12:00:00+00:00")
    assert _submission_date(submission) == "2026-10-27"


def test_submission_date_outside_git_returns_none(tmp_path: Path) -> None:
    assert _submission_date(SimpleNamespace(checkpoint=tmp_path / "model.pt")) is None


def test_submission_date_without_git_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "")
    assert _submission_date(SimpleNamespace(checkpoint=tmp_path / "model.pt")) is None
