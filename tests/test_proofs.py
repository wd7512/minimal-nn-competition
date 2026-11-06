import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from minimal_nn_competition.model.entry import (
    Submission,
    SubmissionConfig,
    SubmissionError,
    materialize_submission,
    parameter_count,
)
from minimal_nn_competition.model.limits import parameter_count_from_dimensions
from proofs.pipeline import BENCHMARK

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides):
    values = {
        "benchmark": BENCHMARK,
        "name": "example",
        "author": "tester",
        "architecture": "mlp",
        "input_size": 4,
        "output_size": 2,
        **overrides,
    }
    return SubmissionConfig(**values)


def test_architecture_table_matches_evaluator():
    from proofs.pipeline import load_architectures
    from proofs.pipeline import parameter_count as proof_parameter_count

    architectures = load_architectures(ROOT)
    assert architectures
    for architecture in architectures:
        expected = parameter_count_from_dimensions(
            architecture.input_size,
            architecture.hidden_sizes,
            architecture.output_size,
        )
        assert architecture.parameters == expected
        assert (
            proof_parameter_count(
                architecture.input_size,
                architecture.hidden_sizes,
                architecture.output_size,
            )
            == expected
        )


def test_generated_lean_is_current():
    from proofs.pipeline import GENERATED_LEAN, load_architectures, render_generated_lean

    expected = render_generated_lean(load_architectures(ROOT))
    assert (ROOT / GENERATED_LEAN).read_text(encoding="utf-8") == expected


@pytest.fixture(scope="module")
def bounds_document():
    from proofs.pipeline import build_bounds

    return build_bounds(ROOT)


def test_bounds_json_is_current(bounds_document):
    from proofs.pipeline import BOUNDS_JSON, render_document

    expected = render_document(bounds_document)
    assert (ROOT / BOUNDS_JSON).read_text(encoding="utf-8") == expected


def test_cartpole_witness(bounds_document):
    witness = bounds_document["evidence"]["empirical_witness"]
    assert witness["kind"] == "empirical_witness"
    assert witness["value"] == 9
    assert witness["score"] == 1.0
    assert witness["passed"] is True
    assert witness["seeds"] == 100
    assert (
        witness["evaluation_sha256"]
        == "c63c576a4e2b74228c282a008d0e45ce2ad63a5bce356a106c874f1c31e9018f"
    )

    rows = json.loads((ROOT / "docs/leaderboard.json").read_text(encoding="utf-8"))
    row = next(
        item
        for item in rows
        if item["benchmark"] == BENCHMARK and item["name"] == "relu-threshold-9"
    )
    assert row["parameter_count"] == 9
    assert row["passed"] is True
    assert row["details"]["evaluation_sha256"] == witness["evaluation_sha256"]


def test_leaderboard_has_no_smaller_cartpole_entry():
    rows = json.loads((ROOT / "docs/leaderboard.json").read_text(encoding="utf-8"))
    cartpole = [row for row in rows if row["benchmark"] == BENCHMARK and row["passed"]]
    assert cartpole
    assert min(row["parameter_count"] for row in cartpole) >= 9
    assert any(row["parameter_count"] == 9 for row in cartpole)


def test_manifest_evidence_classes(bounds_document):
    evidence = bounds_document["evidence"]
    assert set(evidence) == {
        "formal_theorem",
        "computation_certificate",
        "empirical_witness",
    }
    assert evidence["formal_theorem"]["kind"] == "formal_theorem"
    assert evidence["formal_theorem"]["theorem"] == "MinimalNN.cartPole_architecturalMinimum"
    assert evidence["computation_certificate"]["kind"] == "computation_certificate"
    assert evidence["empirical_witness"]["kind"] == "empirical_witness"
    conclusion = bounds_document["conclusion"]
    assert conclusion["status"] == "combined"
    assert conclusion["value"] == 9
    assert conclusion["requires"] == ["formal_theorem", "empirical_witness"]
    assert "Lean-verified" in conclusion["note"]


def test_acceptance_rules():
    assert _config(hidden_sizes=[1]).hidden_sizes == (1,)
    assert parameter_count_from_dimensions(4, (1,), 2) == 9

    invalid = (
        {"input_size": 3},
        {"output_size": 1},
        {"hidden_sizes": [0]},
        {"hidden_sizes": [-1]},
        {"architecture": "linear"},
        {"activation": "tanh"},
        {"input_indices": [0]},
    )
    for overrides in invalid:
        with pytest.raises(ValidationError):
            _config(**overrides)


def test_non_float32_witness_is_rejected(tmp_path):
    directory = tmp_path / "bad"
    directory.mkdir()
    config = {
        "benchmark": BENCHMARK,
        "name": "bad",
        "author": "tester",
        "architecture": "mlp",
        "input_size": 4,
        "output_size": 2,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {"0.weight": torch.zeros((2, 4), dtype=torch.float64), "0.bias": torch.zeros(2)},
        directory / "model.pt",
    )
    with pytest.raises(SubmissionError, match="float32"):
        materialize_submission(Submission.from_directory(directory))


def test_witness_state_dict_uses_formula():
    submission = Submission.from_directory(ROOT / "models/cartpole-gym-v1/relu-threshold-9")
    loaded = materialize_submission(submission)
    config = loaded.config
    assert (config.input_size, config.output_size) == (4, 2)
    assert tuple(config.hidden_sizes) == (1,)
    assert parameter_count(loaded.state_dict) == 9
    assert parameter_count_from_dimensions(4, (1,), 2) == 9


def test_acceptance_contract_is_unchanged():
    from proofs.pipeline import verify_acceptance

    acceptance = verify_acceptance(ROOT)
    assert acceptance["benchmark"] == BENCHMARK
    assert acceptance["architecture"]["input_size"] == 4
    assert acceptance["architecture"]["output_size"] == 2


def test_lean_sources_have_no_placeholders():
    for path in sorted((ROOT / "proofs/lean").rglob("*.lean")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:sorry|admit|native_decide)\b", text), path


def test_lean_toolchain_is_pinned():
    toolchain = (ROOT / "proofs/lean/lean-toolchain").read_text(encoding="utf-8")
    assert toolchain.strip() == "leanprover/lean4:v4.35.0"


def test_check_lean_cli():
    result = subprocess.run(
        [sys.executable, "-m", "proofs", "check-lean"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
