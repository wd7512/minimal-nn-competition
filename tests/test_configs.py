from pathlib import Path

from minimal_nn_competition.benchmarks.registry import VERIFY_BENCHMARKS
from minimal_nn_competition.model.entry import SubmissionConfig, VerifierConfig

ROOT = Path(__file__).resolve().parents[1]


def test_all_checked_in_configs_are_valid() -> None:
    configs = sorted((ROOT / "models").glob("*/*/config.json"))
    assert configs
    for path in configs:
        verifier = path.parent.parent.name in VERIFY_BENCHMARKS
        (VerifierConfig if verifier else SubmissionConfig).from_file(path)
