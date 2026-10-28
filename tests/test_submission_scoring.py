from pathlib import Path

import pytest


def test_score_submission_process_success(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from minimal_nn_competition.submissions.scoring import score_submission_process

    result = score_submission_process(root / "models/snake-simple-8x8/example", tmp_path)
    assert result.passed is not None


def test_score_submission_process_timeout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.scoring import score_submission_process

    with pytest.raises(SubmissionError, match="timed out"):
        score_submission_process(
            root / "models/snake-simple-8x8/example",
            tmp_path,
            timeout=0.001,
        )


def test_score_submission_process_failure(tmp_path: Path) -> None:
    import json

    import torch

    root = tmp_path / "repo"
    root.mkdir()
    (root / "models" / "snake-simple-8x8" / "bad").mkdir(parents=True)
    (root / "models" / "snake-simple-8x8" / "bad" / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "snake-simple-8x8",
                "name": "bad",
                "author": "tester",
                "architecture": "linear",
                "input_size": 24,
                "output_size": 4,
            }
        )
    )
    torch.save(
        {"weight": torch.zeros((3, 7)), "bias": torch.zeros(3)},
        root / "models" / "snake-simple-8x8" / "bad" / "model.pt",
    )
    from minimal_nn_competition.model.entry import SubmissionError
    from minimal_nn_competition.submissions.scoring import score_submission_process

    with pytest.raises(SubmissionError):
        score_submission_process(root / "models/snake-simple-8x8/bad", tmp_path)
