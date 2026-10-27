# Shuffled-label image benchmarks

`mnist-shuffled` and `fashion-mnist-shuffled` score exact accuracy over all 70,000
official images. The shuffled labels are a fixed v1 permutation of the official
labels. These public tasks measure recall on a fixed set; they do not measure
generalisation to new images.

## Frozen contract

Images follow the official order used by `download-mnist` and
`download-fashion-mnist`: all 60,000 training examples, then all 10,000 test
examples. Each image is flattened in row-major order to 784 uint8 values. The raw
IDX files are checksum-pinned in the corresponding benchmark modules. Label vectors
were generated with the issue 191 contract: a fresh NumPy PCG64 generator with seed
191 per source dataset, permuting that dataset's 70,000 original labels. The two
vectors below are committed as raw uint8 files, so scoring does not depend on NumPy's
runtime version.

| Benchmark | Label file | Label SHA-256 | Count per class |
| --- | --- | --- | --- |
| MNIST shuffled | `benchmarks/data/mnist-shuffled-v1.bin` | `ec783f3bee54df90c63abf522174265fa90f01db49e234ce58db58eb09c72719` | `[6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]` |
| Fashion-MNIST shuffled | `benchmarks/data/fashion-mnist-shuffled-v1.bin` | `d290faeac22226f9b8695b363b9d789b6f7138ec5f959d0465da9199c7cf9f68` | `[7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000, 7000]` |

The source file checksums, label generator, independent-uniform comparison,
duplicate-input analysis and pilot training measurements are recorded in
[issue 191's report](shuffled-label-pilot-191.md). The pilot found zero exact
duplicate input groups for either source dataset, so the frozen vectors have no
same-input label conflicts. The majority-class constant baselines are 11.2529% for
MNIST shuffled and 10% for Fashion-MNIST shuffled.

## Download and evaluate

```sh
uv run mnnc download-mnist-shuffled --output data
uv run mnnc download-fashion-mnist-shuffled --output data
```

No shuffled-label example submission is provided because no passing reference
exists. Once a submission directory is available, evaluate it with
`uv run mnnc evaluate mnist-shuffled <submission-directory> --data-path
data/mnist-shuffled.pt` (and replace the benchmark and cache names for
Fashion-MNIST). Evaluation processes all 70,000 examples in batches of 512.
`mnnc collect` accepts `--mnist-shuffled-data-path` and
`--fashion-mnist-shuffled-data-path`.

## Reference feasibility

The issue 191 pilot trained `[784, 64, 10]` MLPs for 20 epochs. It reached 15.6114%
on MNIST shuffled and 11.6229% on Fashion-MNIST shuffled, so neither passed the
100% threshold. Those measurements are a small-model attempt, not a lower bound on
what fits the 10 million parameter and 100 MiB submission limits. No passing
reference checkpoint is published here; a near-limit search and verified passing
artifacts remain necessary before these tracks can be presented as achievable
competitions. The information count `70,000 * log2(10) ≈ 232,535` bits is a
reference calculation, not a proven MLP size lower bound.

The main image leaderboard includes a tested comparison table with true-label and
shuffled-label best scores and parameter counts side by side. It also lists the
constant baselines and the counting-reference caveat. Scores on shuffled tracks
describe exact recall of the public examples and should not be read as image
understanding.
