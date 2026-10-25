from pathlib import Path

import pytest

from minimal_nn_competition import dev_tools


def test_repo_root_is_a_git_root() -> None:
    assert (dev_tools.ROOT / ".git").exists()


def test_delta_messages_have_normal_and_extreme_bands() -> None:
    assert "are you sure" in dev_tools._delta_message(1)
    assert "does not feel" in dev_tools._delta_message(101)
    assert "yay" in dev_tools._delta_message(-1)
    assert "impressively" in dev_tools._delta_message(-101)
    assert dev_tools._delta_message(0) == ""


def test_baseline_is_cleaned_up() -> None:
    try:
        dev_tools._baseline_ref()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    with dev_tools._baseline() as baseline:
        assert (baseline / "pyproject.toml").is_file()
        path = Path(baseline)
    assert not path.exists()


def test_markdown_files_follow_gitignore(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_git(*args, **kwargs):
        calls.append(args)
        return "visible.md"

    monkeypatch.setattr(dev_tools, "_git", fake_git)

    assert dev_tools._markdown_files(tmp_path, respect_gitignore=True) == [tmp_path / "visible.md"]
    assert calls == [("ls-files", "--cached", "--others", "--exclude-standard", "*.md")]
