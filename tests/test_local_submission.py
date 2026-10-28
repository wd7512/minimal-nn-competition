import json
from dataclasses import replace
from pathlib import Path

import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.leaderboard import load_results, save_results
from minimal_nn_competition.model.entry import SubmissionError
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions import local


def _folder(path: Path, benchmark="mnist") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "name": "Display name",
                "author": "tester",
                "architecture": "linear",
                "input_size": 784,
                "output_size": 10,
            }
        )
    )
    (path / "model.pt").write_bytes(b"checkpoint")
    return path


def _result(name="tiny", benchmark="mnist"):
    return BenchmarkResult(name, benchmark, 0.99, True, 7850, {}, name="Display name")


def test_local_report_does_not_write_site(tmp_path, monkeypatch, caplog):
    folder = _folder(tmp_path / "my-model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local, "score_submission_process", lambda *a, **kw: _result())
    monkeypatch.setattr(local, "update_artifacts", lambda *a: pytest.fail("unexpected update"))
    with caplog.at_level("INFO"):
        assert main(["validate-submission", str(folder)]) == 0
    assert '"parameter_count": 7850' in caplog.text
    assert not (tmp_path / "docs").exists()


@pytest.mark.parametrize("parent", ["models/mnist", "mnist"])
def test_path_benchmark_must_match_config(tmp_path, parent):
    folder = _folder(tmp_path / parent / "tiny", benchmark="cartpole")
    with pytest.raises(SubmissionError, match="does not match"):
        local.validate_directory(folder)


def test_unknown_models_benchmark_is_rejected(tmp_path):
    folder = _folder(tmp_path / "models/typo/tiny")
    with pytest.raises(SubmissionError, match="does not match"):
        local.validate_directory(folder)


def test_matching_path_and_arbitrary_location_are_supported(tmp_path):
    for parent in ("models/mnist", "mnist", "somewhere"):
        folder = _folder(tmp_path / parent / "tiny")
        assert local.validate_directory(folder).config.benchmark == "mnist"


def test_invalid_submission_does_not_update(tmp_path, monkeypatch):
    folder = _folder(tmp_path / "tiny")
    (folder / "model.pt").unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local, "score_submission_process", lambda *a, **kw: pytest.fail("scoring"))
    assert main(["validate-submission", str(folder), "--update"]) == 2
    assert not (tmp_path / "docs").exists()


def test_scoring_error_preserves_existing_artifacts(tmp_path, monkeypatch):
    folder = _folder(tmp_path / "tiny")
    output = tmp_path / "docs"
    output.mkdir()
    (output / "index.html").write_text("old site")

    def fail(*args, **kwargs):
        raise SubmissionError("invalid checkpoint")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local, "score_submission_process", fail)
    assert main(["validate-submission", str(folder), "--update"]) == 2
    assert (output / "index.html").read_text() == "old site"


def test_update_preserves_rows_and_uses_folder_identity(tmp_path, monkeypatch):
    folder = _folder(tmp_path / "tiny")
    output = tmp_path / "docs"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    save_results(output / "leaderboard.json", [_result("other"), _result("tiny", "fashion-mnist")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local, "score_submission_process", lambda *a, **kw: _result())
    for _ in range(2):
        assert main(["validate-submission", str(folder), "--update"]) == 0
    assert {(r.benchmark, r.submission) for r in load_results(output / "leaderboard.json")} == {
        ("mnist", "other"),
        ("mnist", "tiny"),
        ("fashion-mnist", "tiny"),
    }
    assert (output / "index.html").is_file()
    assert (output / "keep.txt").read_text() == "keep"


def test_render_error_preserves_entire_site(tmp_path, monkeypatch):
    folder = _folder(tmp_path / "tiny")
    output = tmp_path / "docs"
    output.mkdir()
    save_results(output / "leaderboard.json", [_result("other")])
    before = (output / "leaderboard.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(local, "render_site", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        local.update_artifacts(folder, local.validate_directory(folder), _result(), output)
    assert (output / "leaderboard.json").read_bytes() == before


def test_below_threshold_result_returns_one(tmp_path, monkeypatch):
    folder = _folder(tmp_path / "tiny")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        local, "score_submission_process", lambda *a, **kw: replace(_result(), passed=False)
    )
    assert main(["validate-submission", str(folder)]) == 1


def test_update_records_game_replay(tmp_path):
    folder = Path(__file__).resolve().parents[1] / "models/snake-simple-8x8/hamiltonian-3-h6"
    output = tmp_path / "docs"
    result = _result(folder.name, "snake-simple-8x8")
    local.update_artifacts(folder, local.validate_directory(folder), result, output)
    replays = list((output / "games").glob("*.json"))
    assert len(replays) == 1
    assert json.loads(replays[0].read_text())
