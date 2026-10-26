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
    config = {"name": "example", "author": "tester", **config}
    (directory / "config.json").write_text(json.dumps(config))
    torch.save(state, directory / "model.pt")
    return directory


def test_linear_submission_loads(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
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
            "benchmark": "snake-simple-8x8",
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
            "benchmark": "snake-simple-8x8",
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
    path.write_text(
        '{"benchmark":"snake-simple-8x8","architecture":"cnn","input_size":7,"output_size":4}'
    )
    with pytest.raises(SubmissionError, match="architecture must be one of"):
        SubmissionConfig.from_file(path)


def test_input_indices_rejected_outside_snake(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"mnist","name":"x","author":"tester","architecture":"mlp",'
        '"input_size":784,"output_size":10,"input_indices":[0,1,2]}'
    )
    with pytest.raises(SubmissionError, match="only supported for snake benchmarks"):
        SubmissionConfig.from_file(path)


def test_input_indices_accepted_for_snake(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"snake-simple-8x8","name":"x","author":"tester","architecture":"mlp",'
        '"input_size":3,"output_size":4,"hidden_sizes":[4],"input_indices":[0,1,2]}'
    )
    assert SubmissionConfig.from_file(path).input_indices == (0, 1, 2)


def test_nan_weights_are_rejected(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 7,
            "output_size": 4,
        },
        {"weight": torch.full((4, 7), float("nan")), "bias": torch.zeros(4)},
    )
    with pytest.raises(SubmissionError, match="non-finite"):
        materialize_submission(Submission.from_directory(directory))


def test_inf_weights_are_rejected(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "linear",
            "input_size": 7,
            "output_size": 4,
        },
        {"weight": torch.full((4, 7), float("inf")), "bias": torch.zeros(4)},
    )
    with pytest.raises(SubmissionError, match="non-finite"):
        materialize_submission(Submission.from_directory(directory))


def test_over_parameter_checkpoint_is_rejected(tmp_path: Path) -> None:
    hidden = 1_500_000
    directory = _write_submission(
        tmp_path,
        {
            "benchmark": "snake-simple-8x8",
            "architecture": "mlp",
            "input_size": 3,
            "output_size": 4,
            "hidden_sizes": [hidden],
        },
        {
            "0.weight": torch.zeros((hidden, 3)),
            "0.bias": torch.zeros(hidden),
            "2.weight": torch.zeros((4, hidden)),
            "2.bias": torch.zeros(4),
        },
    )
    with pytest.raises(SubmissionError, match="parameters"):
        materialize_submission(Submission.from_directory(directory))


def test_declared_parameter_cap_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"benchmark":"mnist","name":"too_big","author":"tester","architecture":"mlp",'
        '"input_size":784,"output_size":10,"hidden_sizes":[1000000]}'
    )
    with pytest.raises(SubmissionError, match="parameters"):
        SubmissionConfig.from_file(path)
