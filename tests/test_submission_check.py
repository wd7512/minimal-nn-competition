from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "validate_submission.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_submission", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_submission(
    root: Path,
    *,
    benchmark: str = "mnist",
    name: str = "tiny",
    config: dict[str, object] | None = None,
) -> Path:
    directory = root / "models" / benchmark / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            config
            or {
                "benchmark": benchmark,
                "name": "tiny",
                "author": "tester",
                "architecture": "mlp",
                "input_size": 784,
                "output_size": 10,
                "hidden_sizes": [7],
                "activation": "relu",
            }
        )
    )
    (directory / "model.pt").write_bytes(b"placeholder")
    return directory


def _diff(*paths: str) -> str:
    return "".join(f"A\t{path}\n" for path in paths)


def test_valid_submission_passes(tmp_path: Path) -> None:
    _write_submission(tmp_path)
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    assert _load_script().validate_diff(diff, tmp_path) == []


def test_fashion_mnist_submission_passes(tmp_path: Path) -> None:
    _write_submission(tmp_path, benchmark="fashion-mnist")
    diff = _diff(
        "models/fashion-mnist/tiny/config.json",
        "models/fashion-mnist/tiny/model.pt",
    )
    assert _load_script().validate_diff(diff, tmp_path) == []


def test_code_change_is_rejected(tmp_path: Path) -> None:
    _write_submission(tmp_path)
    diff = _diff(
        "models/mnist/tiny/config.json",
        "models/mnist/tiny/model.pt",
        "src/minimal_nn_competition/__init__.py",
    )
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("only models/" in error for error in errors)


def test_single_file_is_rejected(tmp_path: Path) -> None:
    _write_submission(tmp_path)
    errors = _load_script().validate_diff(_diff("models/mnist/tiny/config.json"), tmp_path)
    assert any("exactly two submission files" in error for error in errors)


def test_benchmark_must_match_directory(tmp_path: Path) -> None:
    directory = _write_submission(
        tmp_path, config={"benchmark": "snake-simple-8x8", "architecture": "mlp"}
    )
    diff = _diff(
        f"models/mnist/{directory.name}/config.json",
        f"models/mnist/{directory.name}/model.pt",
    )
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("benchmark must be 'mnist'" in error for error in errors)


def test_empty_checkpoint_is_rejected(tmp_path: Path) -> None:
    directory = _write_submission(tmp_path)
    (directory / "model.pt").write_bytes(b"")
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("must not be empty" in error for error in errors)


def test_missing_identity_is_rejected(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("name must be a non-empty string" in error for error in errors)
    assert any("author must be a non-empty string" in error for error in errors)


def test_valid_input_indices_pass(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        benchmark="snake-simple-8x8",
        config={
            "benchmark": "snake-simple-8x8",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 5,
            "output_size": 4,
            "hidden_sizes": [7],
            "activation": "relu",
            "input_indices": [0, 1, 2, 4, 7],
        },
    )
    diff = _diff(
        "models/snake-simple-8x8/tiny/config.json",
        "models/snake-simple-8x8/tiny/model.pt",
    )
    assert _load_script().validate_diff(diff, tmp_path) == []


def test_input_indices_must_match_input_size(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 3,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
            "input_indices": [0, 1],
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("length must match input_size" in error for error in errors)


def test_input_indices_must_be_true_or_list(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
            "input_indices": False,
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("must be true or a list" in error for error in errors)


def test_duplicate_input_indices_are_rejected(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 2,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
            "input_indices": [0, 0],
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("must not contain duplicates" in error for error in errors)


def test_description_is_accepted(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "description": "A small MLP trained on raw pixels.",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    assert _load_script().validate_diff(diff, tmp_path) == []


def test_description_must_be_a_string(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "description": 123,
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("description must be a string" in error for error in errors)


def test_over_parameter_submission_is_rejected(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [1_000_001],
            "activation": "relu",
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("parameters" in error for error in errors)


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    _write_submission(
        tmp_path,
        config={
            "benchmark": "mnist",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": [7],
            "activation": "relu",
            "foo": 1,
        },
    )
    diff = _diff("models/mnist/tiny/config.json", "models/mnist/tiny/model.pt")
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("unknown config" in error.lower() for error in errors)


def test_snake_input_indices_outside_zero_to_twenty_three_are_rejected(
    tmp_path: Path,
) -> None:
    _write_submission(
        tmp_path,
        benchmark="snake-simple-8x8",
        config={
            "benchmark": "snake-simple-8x8",
            "name": "tiny",
            "author": "tester",
            "architecture": "mlp",
            "input_size": 1,
            "output_size": 4,
            "hidden_sizes": [7],
            "activation": "relu",
            "input_indices": [24],
        },
    )
    diff = _diff(
        "models/snake-simple-8x8/tiny/config.json", "models/snake-simple-8x8/tiny/model.pt"
    )
    errors = _load_script().validate_diff(diff, tmp_path)
    assert any("input_indices" in error and "0..23" in error for error in errors)
