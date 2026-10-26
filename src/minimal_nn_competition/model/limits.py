"""Shared submission resource limits.

These limits are competition policy for accepted ``linear`` and ``mlp`` state-dict
submissions. They bound declared and on-disk model size without depending on
PyTorch loading or evaluation.
"""

# A generous ceiling above plausible competitive models while preventing an
# entry from declaring an arbitrarily large allocation.
MAX_PARAMETERS = 10_000_000

# Checkpoint files are bounded independently because a malformed file may be
# large relative to its useful tensor contents.
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


def parameter_count_from_dimensions(input_size, hidden_sizes, output_size):
    """Compute parameters from dimensions, before any model is allocated."""
    total = 0
    previous = input_size
    for size in (*tuple(hidden_sizes), output_size):
        total += previous * size + size
        previous = size
    return total
