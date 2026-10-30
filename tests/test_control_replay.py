import subprocess
from dataclasses import replace

import pytest
from test_rl_environments import zero_submission

from minimal_nn_competition.benchmarks.common.control import environment, rollout
from minimal_nn_competition.benchmarks.registry import RL_BENCHMARKS, create_benchmark
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.runner.entry import evaluate_submission
from minimal_nn_competition.submissions.artifacts import _load_replays, _record_replay


@pytest.mark.parametrize("name", RL_BENCHMARKS)
def test_native_video_uses_scoring_trajectory_and_survives_in_place_render(
    tmp_path, monkeypatch, name
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    benchmark = create_benchmark(name, seeds=(0,), max_steps=3)
    submission = zero_submission(tmp_path / "models", name)
    loaded = benchmark.load_submission(submission)
    with environment(name, max_steps=3) as env:
        expected = rollout(env, name, loaded.model, 0)
    with environment(name, render=True, max_steps=3) as env:
        assert rollout(env, name, loaded.model, 0, tmp_path / "test.mp4") == expected
    site = tmp_path / "docs"
    _record_replay(site / "games", "zero", benchmark, submission)
    video = site / "games" / f"{name}--zero.mp4"
    assert video.is_file() and not list((site / "games").glob("*.json"))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    content = video.read_bytes()
    result = evaluate_submission(submission, benchmark)
    result = replace(result, submitted_at="2026-10-30")
    render_site([result], site, _load_replays(site / "games"))
    assert video.read_bytes() == content
    page = (site / "submissions" / f"{name}--zero.html").read_text()
    assert "<video controls playsinline" in page
    assert f"../games/{name}--zero.mp4" in page
    assert "<canvas" not in page and "drawCartpole" not in page
    # Publishing to another directory also carries the media bytes.
    render_site([result], tmp_path / "published", _load_replays(site / "games"))
    assert (tmp_path / "published" / "games" / video.name).read_bytes() == content
