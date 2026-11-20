import copy
import json

import numpy as np
import pytest
import torch

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.verification import interval
from minimal_nn_competition.verification.certificate import (
    CertificateError,
    check_certificate,
    make_certificate,
)
from minimal_nn_competition.verification.model import load_model, to_numpy
from minimal_nn_competition.verification.reference import (
    all_sequences,
    make_reference,
    model_id,
    next_logits,
    plant_violation,
    reference_spec,
)
from minimal_nn_competition.verification.verify import branch_and_bound, oracle

SEED_SPEC = reference_spec(vocab=2, max_len=2, d_model=4, layers=1, heads=1)
SEED_ID = model_id(SEED_SPEC, 0)


@pytest.fixture(scope="module")
def seed_model():
    return to_numpy(make_reference(SEED_SPEC, seed=0))


@pytest.fixture(scope="module")
def seed_cert(seed_model):
    result = branch_and_bound(seed_model, SEED_SPEC, forbidden=1)
    return json.loads(json.dumps(make_certificate(result, SEED_ID, 1)))


def _random_interval(rng, shape):
    lo = rng.normal(size=shape)
    return lo, lo + rng.uniform(0, 1, size=shape)


def _inside(rng, lo, hi):
    return lo + (hi - lo) * rng.uniform(size=lo.shape)


def test_interval_ops_contain_point_results():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lo, hi = _random_interval(rng, (3, 5))
        weight, bias = rng.normal(size=(5, 4)), rng.normal(size=4)
        x = _inside(rng, lo, hi)
        llo, lhi = interval.linear(lo, hi, weight, bias)
        assert np.all((llo <= x @ weight + bias) & (x @ weight + bias <= lhi))
        slo, shi = _random_interval(rng, (6,))
        s = _inside(rng, slo, shi)
        p = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
        plo, phi = interval.softmax(slo, shi)
        assert np.all((plo <= p) & (p <= phi))


def test_singleton_bounds_bracket_torch_forward():
    spec = reference_spec(vocab=3, max_len=3, d_model=8, layers=2, heads=2)
    params = make_reference(spec, seed=1, steps=5)
    weights = to_numpy(params)
    double = {name: value.double() for name, value in params.items()}
    seqs = list(all_sequences(spec))
    exact = next_logits(double, spec, seqs).detach().numpy()
    for seq, row in zip(seqs, exact, strict=True):
        lo, hi = interval.bound_logits(weights, spec, [[t] for t in seq])
        assert np.all((lo <= row) & (row <= hi))
        assert np.all(hi - lo < 1e-6)


def test_parameter_count_matches_tensors():
    spec = reference_spec(vocab=5, max_len=4, d_model=8, layers=2, heads=4)
    params = make_reference(spec, seed=0, steps=1)
    assert spec.parameter_count() == sum(value.numel() for value in params.values())


def test_seed_case_is_proved_and_certificate_checks(seed_model, seed_cert):
    assert seed_cert["answer"] == "PROVED"
    assert check_certificate(seed_model, SEED_SPEC, 1, SEED_ID, seed_cert) == "PROVED"
    truth = oracle(seed_model, SEED_SPEC, 1, SEED_ID)
    assert truth["answer"] == "PROVED"
    assert [row["seq"] for row in truth["inputs"]] == [[], [0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]
    assert all(row["argmax"] == 0 and row["margin"] > 0 for row in truth["inputs"])


def test_planted_model_gives_checked_counterexample():
    params = plant_violation(make_reference(SEED_SPEC, seed=0), SEED_SPEC, 1, (0, 1))
    weights = to_numpy(params)
    result = branch_and_bound(weights, SEED_SPEC, forbidden=1)
    assert result.status == "COUNTEREXAMPLE"
    cert = make_certificate(result, "planted", 1)
    assert check_certificate(weights, SEED_SPEC, 1, "planted", cert) == "COUNTEREXAMPLE"
    truth = oracle(weights, SEED_SPEC, 1, "planted")
    assert truth["answer"] == "COUNTEREXAMPLE"
    assert next(row for row in truth["inputs"] if row["seq"] == [0, 1])["argmax"] == 1


def _tampered(cert, change):
    cert = copy.deepcopy(cert)
    change(cert)
    return cert


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["leaves"].pop(), "cover"),
        (lambda c: c["leaves"].append(c["leaves"][-1]), "overlap|cover"),
        (lambda c: c["leaves"][-1].update(witness=1), "differ"),
        (lambda c: c["leaves"][-1].update(box=[[0], [0, 0]]), "repeat"),
        (lambda c: c.update(model_id="other"), "different model"),
        (lambda c: c.update(answer="COUNTEREXAMPLE", sequence=[1, 1]), "not a strict"),
        (lambda c: c.update(answer="MAYBE"), "answer must be"),
    ],
)
def test_checker_rejects_bad_certificates(seed_model, seed_cert, change, message):
    with pytest.raises(CertificateError, match=message):
        check_certificate(seed_model, SEED_SPEC, 1, SEED_ID, _tampered(seed_cert, change))


def test_checker_rejects_leaf_that_does_not_prove(seed_cert):
    weights = to_numpy(plant_violation(make_reference(SEED_SPEC, 0), SEED_SPEC, 1, (0, 1)))
    with pytest.raises(CertificateError, match="not proved"):
        check_certificate(weights, SEED_SPEC, 1, SEED_ID, seed_cert)


def test_unknown_makes_no_claim(seed_model):
    cert = {"format": 1, "model_id": SEED_ID, "forbidden": 1, "answer": "UNKNOWN"}
    assert check_certificate(seed_model, SEED_SPEC, 1, SEED_ID, cert) == "UNKNOWN"


@pytest.mark.parametrize("seed", range(4))
def test_branch_and_bound_agrees_with_oracle(seed):
    spec = reference_spec(vocab=3, max_len=2, d_model=4, layers=1, heads=2)
    weights = to_numpy(make_reference(spec, seed=seed, steps=3 * seed))
    result = branch_and_bound(weights, spec, forbidden=1)
    assert result.status == oracle(weights, spec, 1, "x")["answer"]


def test_cli_round_trip(tmp_path):
    model_dir, cert = tmp_path / "model", tmp_path / "cert.json"
    assert main(["verify-transformer", "--save-model", str(model_dir), "--cert", str(cert)]) == 0
    assert json.loads(cert.read_text())["answer"] == "PROVED"
    assert main(["check-certificate", str(model_dir), str(cert)]) == 0
    spec, params, name, forbidden = load_model(model_dir)
    assert (spec, name, forbidden) == (SEED_SPEC, SEED_ID, 1)
    assert all(value.dtype == torch.float32 for value in params.values())
    bad = json.loads(cert.read_text())
    bad["leaves"].pop()
    cert.write_text(json.dumps(bad))
    assert main(["check-certificate", str(model_dir), str(cert)]) == 1
