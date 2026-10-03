# DiscrimAlign

DiscrimAlign provides a ready-to-run miRNA sequence-pair inference workflow backed by bundled trained models, plus the research code used to train and evaluate those models. The main user-facing path is: provide a CSV of sequence pairs, choose a bundled miRNA model, and receive probabilities plus human-readable alignments.

## Repository structure

```text
src/                         Core DiscrimAlign training and inference code
tests/                       Unit and integration tests (pytest)
examples/mirna_pairs.csv       Ready-to-run inference input example
case_study_for_mirna/         Bundled miRNA trained models and evaluation workflows
benchmarks/                   Backend benchmark script
TODO-nwgrad.md                Follow-ups to the nwgrad backend
Simulation experiments.ipynb  Simulation experiments for the manuscript
pyproject.toml                Project environment managed by uv
```

## Requirements

- Python `>=3.10`
- `uv` for environment management
- `nwgrad` 0.5.0 or later, the default alignment backend. `uv sync` installs it as a binary wheel. Building it from source needs a C++20 compiler that provides `<experimental/simd>`, such as GCC; on macOS use Homebrew GCC (`CC=gcc-16 CXX=g++-16`), since Apple's clang does not provide it.
- JupyterLab or VS Code notebook support for running `Simulation experiments.ipynb`

The repository uses a single project environment managed by `uv`. This environment includes the scientific Python dependencies, JupyterLab, an IPython kernel for notebooks, and `miRBench` for the miRNA case-study dataset interface.

## Installing `uv`

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal and confirm that `uv` is available:

```powershell
uv --version
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal and confirm that `uv` is available:

```bash
uv --version
```

## Project environment

Create the project environment from the repository root:

```bash
uv sync
```

All project commands are run through this environment with `uv run`.

## miRNA Inference Quickstart

Use this workflow when you want predictions from the bundled trained miRNA models without retraining.

### 1. Inspect the Example Input

A ready-to-run CSV is provided at:

```text
examples/mirna_pairs.csv
```

It uses this schema:

```csv
id,sequence_a,sequence_b
example_positive_like,AUGCUA,AUGGUA
example_short,CUGA,CUGU
```

Required columns:

- `sequence_a`: first miRNA/RNA sequence
- `sequence_b`: second miRNA/RNA sequence
- any extra columns, such as `id`, are preserved in the output

### 2. Run a Bundled Model

Two trained miRNA model aliases are available:

- `manakov`: `case_study_for_mirna/trained_models/manakov_best_model.pkl`
- `hejret`: `case_study_for_mirna/trained_models/hejret_best_model.pkl`

Run inference:

```bash
uv run python -m src.infer \
  --model manakov \
  --input examples/mirna_pairs.csv \
  --output predictions_manakov.csv
```

Or use the Hejret-trained model:

```bash
uv run python -m src.infer \
  --model hejret \
  --input examples/mirna_pairs.csv \
  --output predictions_hejret.csv
```

### 3. Read the Output

The output CSV includes:

- `probability`: logistic model probability for the positive class
- `alignment_score`: score assigned by the fitted aligner
- `aligned_sequence_a`, `alignment_marks`, `aligned_sequence_b`: readable alignment
- `operations`: per-position `match`, `mismatch`, or `gap`
- `normalized_sequence_a`, `normalized_sequence_b`: sequences actually scored by the model

By default, `--normalize auto` converts `U`/`T` to match the trained model alphabet. Use `--normalize none` only if you want to disable this behavior.

### 4. Use Your Own CSV Columns

If your input columns have different names, pass them explicitly:

```bash
uv run python -m src.infer \
  --model manakov \
  --input my_pairs.csv \
  --output my_predictions.csv \
  --seq-a-column mirna \
  --seq-b-column target
```

## Simulation experiments

The notebook

```text
Simulation experiments.ipynb
```

contains the simulation experiments associated with the manuscript and is the primary reproducibility material alongside the implementation in `src/`.

Open the notebook with JupyterLab:

```bash
uv run jupyter lab "Simulation experiments.ipynb"
```

### VS Code

With the Python and Jupyter extensions installed, open `Simulation experiments.ipynb` and select the kernel associated with the local `.venv/` environment.

## Training API

Use `discrimalign` directly when you want to fit a new model instead of using the bundled miRNA models.

```python
from src.discrimalign import discrimalign
from src.optimization import create_powerstep

seqlistA = ["AUGCUA", "CUGA"]
seqlistB = ["AUGGUA", "CUGU"]
labels = [1, 0]

result = discrimalign(
    seqlistA=seqlistA,
    seqlistB=seqlistB,
    labels=labels,
    aligner_mode="local",
    gap_mode="affine",
    substitution_mode="symmetric",
    stepfunction=create_powerstep(1e-5),
)

print(result["final_loglik"])
print(result["alpha"])
```

The returned object contains the fitted aligner, learned alignment parameters, intercept, final log-likelihood, and optimization trajectories.
If `stepfunction` is omitted, `discrimalign` uses a conservative default power step with scale `1e-4`.

For inference on new sequence pairs, use `predict_pairs` with the fitted result:

```python
from src import predict_pairs

rows = predict_pairs(
    seqlistA=["AUGCUA"],
    seqlistB=["AUGGUA"],
    model=result,
)

print(rows[0]["probability"])
print(rows[0]["aligned_sequence_a"])
print(rows[0]["alignment_marks"])
print(rows[0]["aligned_sequence_b"])
```

Each inference row is a plain dictionary with the input sequences, normalized sequences, alignment score, logistic probability, aligned strings, match markers, and per-position operations (`match`, `mismatch`, or `gap`). By default, inference normalizes `U`/`T` automatically to match the fitted model alphabet; pass `--normalize none` in the CLI to disable this.

To persist a fitted model and run CSV inference later:

```python
from src import save_model

save_model(result, "model.pkl")
```

Run inference with a custom saved model:

```bash
uv run python -m src.infer --model model.pkl --input examples/mirna_pairs.csv --output predictions.csv
```

`stepfunction` maps the iteration number to a step size; it defaults to `create_powerstep(1e-4)`. `src.optimization` provides `create_powerstep` and `create_constant_step`.

`num_threads=0`, the default, chooses the thread count automatically: all logical cores with the nwgrad backend, and one thread with the Biopython backend, whose threads contend for Python's global interpreter lock and only slow it down. Any other value is used as given. For long sequences such as full-length proteins, the number of physical cores can be faster than all logical cores; pass it explicitly.

`nwgrad_fill` selects nwgrad's vectorized DP fill: `"striped"` (default), `"rowwise"` or `"interpair"`. All three give the same scores, gradients and fit, bit for bit; only the speed differs. On short pairs such as miRNA-target sites the default is the slowest: on all 2.5 million Manakov training pairs (local/affine/general, 300 iterations, 12 threads on an i5-12500), the fit took 1072 s with `"striped"`, 469 s with `"rowwise"` and 291 s with `"interpair"`, which aligns several pairs at once, one per vector lane. The latter two need an nwgrad with `SeqPairBatch.fill` (currently nwgrad `main`, unreleased); on long sequences such as proteins, keep the default.

### Intercept fit

At every iteration the intercept α is refitted to the current alignment scores. `alpha_solver` selects how:

- `"safeguarded_newton"` (default): Newton's method on dL/dα, which is strictly decreasing in α, so its root is the unique optimum. Plain Newton steps are used while they are small, as they are when α changes little between iterations; otherwise the root is bracketed and Newton steps are combined with bisection. The result is exact to rounding from any starting value.
- `"bfgs"`: the previous `scipy.optimize.minimize` fit. When the optimum moves far between iterations, as with `subgradient_scale=1` on large datasets, it can stop far from the optimum.

On all 2.5 million Manakov training pairs, 300 iterations took 18.4 minutes with the default against 29.6 minutes with `"bfgs"`, and gave the same fit.

### Backends

`backend` selects where alignment scores and subgradients come from:

- `"nwgrad"` (default): the [nwgrad](https://github.com/michalsta/nwgrad) C++ library aligns all pairs in parallel and returns each alignment's score and gradient in one pass. The pairs are encoded once and reused across iterations.
- `"biopython"`: Biopython's `PairwiseAligner`, with the gradient counted in Python from the alignment strings.

Both backends fit the same model and return the same kinds of objects: the returned `aligner` and `alignments` are Biopython objects with either backend. With `"nwgrad"` and `return_alignments=True`, the final alignments are computed with Biopython once at the end; pass `return_alignments=False` to skip that pass.

The backends agree up to tie-breaking. When two alignments of a pair score exactly the same but use different substitutions or gaps, the backends may pick different ones. Both are valid subgradients, but the fits then drift apart. Exact ties are common with integer-valued scores, such as the default baseline aligner, and with fitted substitution matrices that contain exactly equal entries, such as the zeros a ridge fit assigns to substitutions that never occur in the data. Away from ties, the two backends follow the same trajectory up to floating-point rounding.

With `"nwgrad"`, the initial estimate is fitted on nwgrad's own alignments of the baseline. A `baseline_aligner` must then be expressible in DiscrimAlign's model: the same gap scores for both sequences and for internal and end gaps, no wildcard, and `local` or `global` mode. Otherwise a `ValueError` explains what is unsupported.

On the x86-64 machines tested, one fitting iteration on 10,000 miRNA-sized pairs (22 × 50 nt, local alignment, affine gaps, full substitution matrix) was 13–22× faster with nwgrad on one thread than with Biopython, and 77–212× faster with all cores. On 300-residue proteins, where Biopython's C alignment is more competitive, it was 3–5× faster on one thread and 26–108× with all cores. See `benchmarks/` to measure your own machine.

### Behavior changes

Compared with earlier versions of DiscrimAlign:

- Gap scores are kept at or below `-1e-4`, both at the start and after every step. A positive gap score rewards gaps, which makes local alignment ill-posed: Biopython's local mode gives inconsistent answers for it, and the two backends would disagree.
- Subgradient gap counts are fixed for alignments that switch directly between a gap in one sequence and a gap in the other. Each switch now counts as a new gap opening, as Biopython scores it; it was previously counted as an extension.
- In `symmetric` and `general` substitution mode with linear gaps, the initial estimate fits one coefficient on the number of gap columns. It previously added the gap-open and gap-extend coefficients of an affine fit.
- Empty sequences raise a `ValueError` before any alignment work.
- The default backend is nwgrad, and `num_threads` defaults to automatic.
- The intercept α is fitted exactly by a safeguarded Newton method (`alpha_solver="safeguarded_newton"`); the previous BFGS fit is available as `alpha_solver="bfgs"`. With labels of only one class, fitting α now raises a `ValueError`, since the likelihood then has no finite maximum.

## Running tests

`uv sync` installs `pytest` with the default `dev` dependency group. From the repository root:

```bash
uv run pytest                  # full suite, about a minute
uv run pytest -m "not slow"    # skip the end-to-end learning runs
```

Tests marked `xfail(strict=True)` document known, accepted differences or bugs; they start failing once the behavior changes, and the marker should then be removed.

## Benchmarks

`benchmarks/bench_backends.py` times both backends on the current machine: one fitting iteration split into its parts, at several thread counts, plus the initial estimate, for miRNA-sized and protein-sized workloads.

```bash
uv run python benchmarks/bench_backends.py --quick        # a few minutes
uv run python benchmarks/bench_backends.py                # full size
uv run python benchmarks/bench_backends.py --fingerprint  # hashes to compare machines
```

The inputs are generated with Python's own random number generator, which gives the same numbers on every platform, so `--fingerprint` output from different machines can be compared directly. nwgrad's results are meant to be bit-identical across CPUs and instruction sets.

## miRNA case study

The miRNA case study, trained models, and instructions for reproducing the manuscript AUPRC metrics are documented in:

```text
case_study_for_mirna/README.md
```
