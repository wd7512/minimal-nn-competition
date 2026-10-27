import importlib.util
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pilot_shuffled_labels.py"
_SPEC = importlib.util.spec_from_file_location("pilot_shuffled_labels", _SCRIPT)
_PILOT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PILOT)
_duplicate_groups = _PILOT._duplicate_groups
make_labels = _PILOT.make_labels


def test_make_labels_is_reproducible_and_preserves_permuted_counts():
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 9])

    shuffled, uniform = make_labels(labels, seed=191)
    repeated_shuffled, repeated_uniform = make_labels(labels, seed=191)

    assert torch.equal(shuffled, repeated_shuffled)
    assert torch.equal(uniform, repeated_uniform)
    assert torch.equal(torch.bincount(shuffled), torch.bincount(labels))
    assert uniform.min() >= 0
    assert uniform.max() < 10


def test_duplicate_group_count_uses_exact_pixel_bytes():
    inputs = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]], dtype=torch.uint8)

    assert _duplicate_groups(inputs) == 2
