from pathlib import Path

import pytest

from minimal_nn_competition.submissions import artifacts


def _site(root: Path, content: str):
    root.mkdir()
    for name in artifacts._SITE_FILES:
        (root / name).write_text(content)
    for name in artifacts._SITE_DIRS:
        (root / name).mkdir()
    (root / "games" / "known.json").write_text(content)


def test_install_preserves_unrendered_files(tmp_path):
    output, rendered = tmp_path / "docs", tmp_path / "rendered"
    _site(output, "old")
    _site(rendered, "new")
    (output / "games" / "notes.json").write_text("notes")
    (output / "submissions" / "notes.txt").write_text("notes")
    artifacts.install_site(rendered, output)
    assert (output / "games" / "known.json").read_text() == "new"
    assert (output / "games" / "notes.json").read_text() == "notes"
    assert (output / "submissions" / "notes.txt").read_text() == "notes"


def test_install_failure_restores_previous_site(tmp_path, monkeypatch):
    output, rendered = tmp_path / "docs", tmp_path / "rendered"
    _site(output, "old")
    _site(rendered, "new")
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    real_replace = artifacts.os.replace

    def fail(source, target):
        if (
            Path(source).parent.name.startswith(".mnnc-stage-")
            and Path(target).name == "index.html"
        ):
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        artifacts.install_site(rendered, output)
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert after == before
    assert not list(output.glob(".mnnc-*"))


def test_failed_restore_retains_recovery_backup(tmp_path, monkeypatch, caplog):
    output, rendered = tmp_path / "docs", tmp_path / "rendered"
    _site(output, "old")
    _site(rendered, "new")
    real_replace = artifacts.os.replace

    def fail(source, target):
        if Path(source).parent.name.startswith((".mnnc-stage-", ".mnnc-backup-")):
            raise OSError("disk failure")
        real_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        artifacts.install_site(rendered, output)
    backups = list(output.glob(".mnnc-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "leaderboard.json").read_text() == "old"
    assert f"backup retained at {backups[0]}" in caplog.text
