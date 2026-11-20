import json
import time
from pathlib import Path

import numpy as np
import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.leaderboard import rank_results
from minimal_nn_competition.model.entry import Submission, SubmissionError, VerifierConfig
from minimal_nn_competition.runner.entry import BenchmarkResult
from minimal_nn_competition.submissions.validation import validate_config, validate_diff
from minimal_nn_competition.verification import interval
from minimal_nn_competition.verification.certificate import CertificateError, check_answer
from minimal_nn_competition.verification.forward import all_sequences, gaps, logits
from minimal_nn_competition.verification.instances import make_instances, random_params
from minimal_nn_competition.verification.model import ModelSpec, load_instance, save_instance
from minimal_nn_competition.verification.runner import run_evaluation
from minimal_nn_competition.verification.verify import branch_and_bound

ROOT = Path(__file__).resolve().parents[1]
LEN0 = ROOT / "models/verify-transformer-len0/example-interval"
LEN4 = ROOT / "models/verify-transformer-len4/example-enumerate"
SPEC4 = ModelSpec(vocab=2, max_len=4, d_model=8, layers=2, heads=2, d_mlp=32)


@pytest.fixture(scope="module")
def instances():
    return list(make_instances(SPEC4, seed=5, count=10))


def _answer(instance, spec=SPEC4):
    answer = branch_and_bound(instance.params, spec, 1, instance.tau)
    return {"id": instance.instance_id} | answer


def _check(instance, answer, spec=SPEC4):
    return check_answer(instance.params, spec, 1, instance.tau, instance.instance_id, answer)


def test_interval_ops_contain_point_results():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lo = rng.normal(size=(3, 5))
        hi = lo + rng.uniform(0, 1, size=lo.shape)
        weight, bias = rng.normal(size=(5, 4)), rng.normal(size=4)
        x = lo + (hi - lo) * rng.uniform(size=lo.shape)
        llo, lhi = interval.linear(lo, hi, weight, bias)
        assert np.all((llo <= x @ weight + bias) & (x @ weight + bias <= lhi))
        slo = rng.normal(size=6)
        shi = slo + rng.uniform(0, 1, size=6)
        s = slo + (shi - slo) * rng.uniform(size=6)
        p = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
        plo, phi = interval.softmax(slo, shi)
        assert np.all((plo <= p) & (p <= phi))


def test_interval_bounds_bracket_forward_pass():
    spec = ModelSpec(vocab=3, max_len=3, d_model=8, layers=2, heads=2, d_mlp=16)
    params = random_params(spec, np.random.default_rng(1))
    for seq, row in zip(
        all_sequences(spec), logits(params, spec, all_sequences(spec)), strict=True
    ):
        lo, hi = interval.bound_logits(params, spec, [[t] for t in seq])
        assert np.all((lo <= row) & (row <= hi))
    full = [list(range(spec.vocab))] * spec.max_len
    lo, hi = interval.bound_logits(params, spec, full)
    rows = logits(params, spec, all_sequences(spec))
    assert np.all((lo <= rows.min(axis=0)) & (rows.max(axis=0) <= hi))


def test_parameter_count_matches_weights():
    params = random_params(SPEC4, np.random.default_rng(0))
    assert SPEC4.parameter_count() == sum(value.size for value in params.values())


def test_instances_are_deterministic_and_labelled(instances):
    again = list(make_instances(SPEC4, seed=5, count=10))
    for first, second in zip(instances, again, strict=True):
        assert first.instance_id == second.instance_id and first.sound == second.sound
        assert all(np.array_equal(first.params[k], second.params[k]) for k in first.params)
    assert [i.instance_id for i in instances] == [f"{n:03d}" for n in range(10)]
    assert sum(i.sound for i in instances) == 5
    for instance in instances:
        gap = gaps(logits(instance.params, SPEC4, all_sequences(SPEC4)), 1)
        assert np.all(np.abs(gap) >= instance.tau)
        violations = [list(s) for s, g in zip(all_sequences(SPEC4), gap, strict=True) if g > 0]
        assert violations == instance.violations
        assert (not violations) if instance.sound else 1 <= len(violations) <= 3


def test_length_zero_has_one_input():
    spec = ModelSpec(vocab=2, max_len=0, d_model=4, layers=1, heads=1, d_mlp=16)
    batch = list(make_instances(spec, seed=0, count=4))
    assert [len(i.violations) for i in batch if not i.sound] == [1, 1]
    for instance in batch:
        assert _check(instance, _answer(instance, spec), spec) in ("PROVED", "COUNTEREXAMPLE")


def test_reference_verifier_answers_are_accepted(instances):
    for instance in instances:
        kind = _check(instance, _answer(instance))
        assert kind == ("PROVED" if instance.sound else "COUNTEREXAMPLE")


def _first(instances, sound):
    return next(i for i in instances if i.sound == sound)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a["leaves"].pop(), "cover"),
        (lambda a: a["leaves"].append(a["leaves"][-1]), "overlap|cover"),
        (lambda a: a["leaves"][-1].update(witness=1), "differ"),
        (lambda a: a["leaves"][-1].update(box=[[0], [0, 0], [0], [0]]), "repeat"),
        (lambda a: a["leaves"][-1].update(box=[[0, 1]] * 4, method="point"), "single"),
        (lambda a: a.update(id="other"), "id"),
        (lambda a: a.update(answer="COUNTEREXAMPLE", sequence=[1, 1, 1, 1]), "not a counter"),
        (lambda a: a.update(answer="MAYBE"), "answer must be"),
    ],
)
def test_checker_rejects_bad_answers(instances, change, message):
    instance = _first(instances, sound=True)
    answer = _answer(instance)
    change(answer)
    with pytest.raises(CertificateError, match=message):
        _check(instance, answer)


def test_checker_rejects_proof_of_unsafe_network(instances):
    sound, unsafe = _first(instances, True), _first(instances, False)
    answer = _answer(sound) | {"id": unsafe.instance_id}
    with pytest.raises(CertificateError, match="fails|does not hold"):
        _check(unsafe, answer)


def test_instance_files_round_trip(tmp_path, instances):
    instance = instances[0]
    save_instance(tmp_path, SPEC4, instance.params, instance.instance_id, 1, instance.tau)
    spec, params, instance_id, forbidden, tau = load_instance(tmp_path)
    assert (spec, instance_id, forbidden, tau) == (SPEC4, instance.instance_id, 1, instance.tau)
    assert all(np.array_equal(params[k], instance.params[k]) for k in params)


def _write_verifier(tmp_path, body):
    path = tmp_path / "verifier.py"
    path.write_text("import json, sys, time\nfrom pathlib import Path\n" + body)
    return path


def test_runner_grades_wrong_and_partial_answers(tmp_path):
    claims_proof = _write_verifier(
        tmp_path,
        "for d in Path(sys.argv[1]).iterdir():\n"
        "    answer = {'id': d.name, 'answer': 'PROVED', 'leaves': [{'box': [[0, 1]] * 4, 'witness': 0}]}\n"
        "    (Path(sys.argv[2]) / f'{d.name}.json').write_text(json.dumps(answer))\n",
    )
    result = run_evaluation(claims_proof, SPEC4, seed=3, count=4)
    assert (result.solved, result.wrong) == (0, 4)
    slow = _write_verifier(
        tmp_path,
        "first = sorted(Path(sys.argv[1]).iterdir())[0].name\n"
        "(Path(sys.argv[2]) / f'{first}.json').write_text(json.dumps({'id': first, 'answer': 'UNKNOWN'}))\n"
        "time.sleep(30)\n",
    )
    result = run_evaluation(slow, SPEC4, seed=3, count=4, time_limit=2)
    assert result.timed_out and (result.unsolved, result.wrong) == (4, 0)


def test_example_submissions_pass_full_evaluation_quickly():
    # LEN0 at its claimed size; LEN4 claims 25M parameters, so test its verifier on SPEC4.
    start = time.monotonic()
    submission = Submission.from_directory(LEN0)
    result = run_evaluation(submission.checkpoint, submission.config.model_spec(), seed=9)
    assert (result.solved, result.wrong) == (100, 0)
    assert time.monotonic() - start < 30
    result = run_evaluation(Submission.from_directory(LEN4).checkpoint, SPEC4, seed=9)
    assert (result.solved, result.wrong) == (100, 0)


def test_evaluate_cli_reports_parameters(capsys):
    assert main(["evaluate", "verify-transformer-len0", str(LEN0), "--seed", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] and result["details"]["solved"] == 100
    assert result["parameter_count"] == Submission.from_directory(LEN0).config.parameter_count()


def test_make_instances_and_check_answer_cli(tmp_path):
    output = tmp_path / "practice"
    args = ["make-instances", "verify-transformer-len4", str(output), "--count", "2", "--seed", "4"]
    assert main(args) == 0
    labels = json.loads((output / "labels.json").read_text())
    instance_dir = output / "instances" / "000"
    spec, params, instance_id, _, tau = load_instance(instance_dir)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"id": instance_id} | branch_and_bound(params, spec, 1, tau)))
    assert main(["check-answer", str(instance_dir), str(answer)]) == 0
    expected = "PROVED" if labels[instance_id]["sound"] else "COUNTEREXAMPLE"
    assert json.loads(answer.read_text())["answer"] == expected


def test_verification_ranks_larger_models_first():
    def entry(name, params, passed=True):
        details = {"solved": 100 if passed else 50, "instances": 100}
        return BenchmarkResult(
            name,
            "verify-transformer-len4",
            1.0 if passed else 0.5,
            passed,
            params,
            details,
            name,
            "author",
            "2026-01-01",
        )

    ranked = rank_results([entry("small", 300), entry("big", 9000), entry("fail", 99999, False)])
    assert [r.name for r in ranked] == ["big", "small", "fail"]


def test_verifier_config_validation(tmp_path):
    raw = json.loads((LEN0 / "config.json").read_text())
    assert VerifierConfig.model_validate(raw).parameter_count() == 236
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
