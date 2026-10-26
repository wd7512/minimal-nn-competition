"""Shared submission resource limits.

These are competition policy values for ``linear`` and ``mlp`` state-dict
submissions. They bound the declared and on-disk model size without depending
on PyTorch loading or evaluation. They are distinct: parameter count measures
the model's computational size, while checkpoint size limits the input file.
One cannot safely be derived from the other without assumptions about tensor
data types and file overhead.
"""

# Generous relative to expected competitions submissions: the largest published
# model in the repository is below one million parameters, and competitive
# targets are orders of magnitude smaller. This ceiling gives experimentation
# room while preventing an entry from declaring an arbitrarily large allocation.
MAX_PARAMETERS = 10_000_000

# 100 MiB covers a checkpoint with 10 million 32-bit parameters plus typical
# state-dict overhead, while preventing submissions from forcing gigabyte file
# sizes before evaluation. It is a preflight limit, not a complete memory quota.
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


def parameter_count_from_dimensions(input_size, hidden_sizes, output_size):
    """Compute parameters from dimensions, before any model is allocated."""
    total = 0
    previous = input_size
    for size in (*tuple(hidden_sizes), output_size):
        total += previous * size + size
        previous = size
    return total
