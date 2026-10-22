import json
from pathlib import Path

import pytest
import torch

from minimal_nn_competition.model.entry import (
    Submission,
    SubmissionConfig,
    SubmissionError,
    materialize_submission,
)


def _write_submission(
    tmp_path: Path, config: dict[str, object], state: dict[str, torch.Tensor]
) -> Path:
    directory = tmp_path / "example"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    torch.save(state, directory / "model.pt")
    return directory


def test_linear_submission_loads(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 7,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 7)), "bias": torch.zeros(4)},
    )
    loaded = materialize_submission(Submission.from_directory(directory))
    assert loaded.model(torch.zeros((2, 7))).shape == (2, 4)


def test_mlp_submission_loads(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "mlp",
            "input_size": 7,
            "output_size": 4,
            "hidden_sizes": [8],
            "activation": "relu",
        },
        {
            "0.weight": torch.zeros((8, 7)),
            "0.bias": torch.zeros(8),
            "2.weight": torch.zeros((4, 8)),
            "2.bias": torch.zeros(4),
        },
    )
    loaded = materialize_submission(Submission.from_directory(directory))
    assert loaded.model(torch.zeros((1, 7))).shape == (1, 4)


def test_mlp_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "mlp",
            "input_size": 7,
            "output_size": 4,
            "hidden_sizes": [8],
        },
        {"0.weight": torch.zeros((9, 7)), "0.bias": torch.zeros(9)},
    )
    with pytest.raises(SubmissionError, match="does not match architecture"):
        materialize_submission(Submission.from_directory(directory))


def test_unknown_architecture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"benchmark":"snake","architecture":"cnn","input_size":7,"output_size":4}')
    with pytest.raises(SubmissionError, match="architecture must be one of"):
        SubmissionConfig.from_file(path)
