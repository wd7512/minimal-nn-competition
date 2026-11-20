import json
from pathlib import Path

import numpy as np
import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.model.entry import Submission, SubmissionError, VerifierConfig
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions.validation import validate_config, validate_diff
from minimal_nn_competition.verification.forward import all_sequences, gaps, logits
from minimal_nn_competition.verification.instances import TAU, make_instances, random_params
from minimal_nn_competition.verification.model import ModelSpec, load_instance, save_instance
from minimal_nn_competition.verification.runner import grade, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
LEN0 = ROOT / "models/verify-transformer-len0/example-enumerate"
LEN4 = ROOT / "models/verify-transformer-len4/example-enumerate"
SPEC4 = ModelSpec(vocab=2, max_len=4, d_model=8, layers=2, heads=2, d_mlp=32)


def test_parameter_count_matches_weights():
    params = random_params(SPEC4, np.random.default_rng(0))
    assert SPEC4.parameter_count() == sum(value.size for value in params.values())


def test_instances_are_deterministic_and_labelled():
    batch = list(make_instances(SPEC4, seed=5, count=10))
    again = list(make_instances(SPEC4, seed=5, count=10))
    assert [i.instance_id for i in batch] == [f"{n:03d}" for n in range(10)]
    assert sum(not i.violations for i in batch) == 5
    for first, second in zip(batch, again, strict=True):
        assert all(np.array_equal(first.params[k], second.params[k]) for k in first.params)
        gap = gaps(logits(first.params, SPEC4, all_sequences(SPEC4)), 1)
        assert np.all(np.abs(gap) >= TAU)
        assert first.violations == [
            list(s) for s, g in zip(all_sequences(SPEC4), gap, strict=True) if g > 0
        ]
        assert len(first.violations) <= 3


def test_length_zero_has_one_input():
    spec = ModelSpec(vocab=2, max_len=0, d_model=4, layers=1, heads=1, d_mlp=16)
    assert sorted(i.violations for i in make_instances(spec, seed=0, count=4)) == [
        [],
        [],
        [[]],
        [[]],
    ]


def test_instance_files_round_trip(tmp_path):
    instance = next(make_instances(SPEC4, seed=1, count=2))
    save_instance(tmp_path, SPEC4, instance.params, instance.instance_id, 1)
    spec, params, instance_id, forbidden = load_instance(tmp_path)
    assert (spec, instance_id, forbidden) == (SPEC4, instance.instance_id, 1)
    assert all(np.array_equal(params[k], instance.params[k]) for k in params)


@pytest.mark.parametrize(
    ("violations", "answer", "outcome"),
    [
        ([], {"answer": "PROVED"}, "solved"),
        ([[0, 1, 1, 0]], {"answer": "PROVED"}, "wrong"),
        ([[0, 1, 1, 0]], {"answer": "COUNTEREXAMPLE", "sequence": [0, 1, 1, 0]}, "solved"),
        ([[0, 1, 1, 0]], {"answer": "COUNTEREXAMPLE", "sequence": [1, 1, 1, 1]}, "wrong"),
        ([], {"answer": "COUNTEREXAMPLE", "sequence": [0, 0, 0, 0]}, "wrong"),
        ([], {"answer": "UNKNOWN"}, "unsolved"),
        ([], None, "unsolved"),
    ],
)
def test_grade(tmp_path, violations, answer, outcome):
    path = tmp_path / "answer.json"
    if answer is not None:
        path.write_text(json.dumps(answer))
    assert grade(violations, path) == outcome


def test_time_limit_keeps_written_answers(tmp_path):
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "import json, sys, time\nfrom pathlib import Path\n"
        "first = sorted(Path(sys.argv[1]).iterdir())[0].name\n"
        "(Path(sys.argv[2]) / f'{first}.json').write_text(json.dumps({'answer': 'PROVED'}))\n"
        "time.sleep(30)\n"
    )
    result = run_evaluation(verifier, SPEC4, seed=3, count=4, time_limit=2)
    assert result.solved + result.wrong == 1 and result.unsolved == 3


def test_example_verifier_solves_every_network():
    for spec in (Submission.from_directory(LEN0).config.model_spec(), SPEC4):
        result = run_evaluation(LEN4 / "verifier.py", spec, seed=9)
        assert (result.solved, result.wrong) == (100, 0)


def test_evaluate_cli_reports_parameters(capsys):
    assert main(["evaluate", "verify-transformer-len0", str(LEN0), "--seed", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] and result["details"]["solved"] == 100
    assert result["parameter_count"] == 236


def test_verification_ranks_larger_models_first():
    def entry(name, params, passed=True):
        details = {"solved": 100 if passed else 50, "instances": 100}
        score = 1.0 if passed else 0.5
        return BenchmarkResult(
            name, "verify-transformer-len4", score, passed, params, details, name, "a", "2026-01-01"
        )

    ranked = rank_results([entry("small", 300), entry("big", 9000), entry("fail", 99999, False)])
    assert [r.name for r in ranked] == ["big", "small", "fail"]


def test_verifier_config_validation(tmp_path):
    raw = json.loads((LEN0 / "config.json").read_text())
    assert validate_config(raw, Path("models/verify-transformer-len0/x")) == []
    assert validate_config(raw | {"heads": 3}, Path("models/verify-transformer-len0/x"))
    assert validate_config(raw, Path("models/verify-transformer-len4/x"))
    (tmp_path / "config.json").write_text(json.dumps(raw | {"layers": 0}))
    with pytest.raises(SubmissionError):
        VerifierConfig.from_file(tmp_path / "config.json")


def test_submission_check_accepts_verifier_files(tmp_path):
    directory = tmp_path / "models/verify-transformer-len0/mine"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text((LEN0 / "config.json").read_text())
    (directory / "verifier.py").write_text("pass\n")
    base = "models/verify-transformer-len0/mine"
    assert validate_diff(f"A\t{base}/config.json\nA\t{base}/verifier.py", tmp_path) == []
    errors = validate_diff(f"A\t{base}/config.json\nA\t{base}/model.pt", tmp_path)
    assert any("verifier.py" in error for error in errors)
