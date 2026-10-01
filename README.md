# DiscrimAlign

DiscrimAlign is a research codebase for discriminatively learning alignment parameters from labelled pairs of biological sequences. The repository contains the core implementation of the method, the simulation experiments used in the manuscript, a stable `uv` environment, and a manuscript-aligned miRNA case study.

## Repository structure

```text
src/                         Core DiscrimAlign implementation
tests/                       Unit and integration tests (pytest)
benchmarks/                   Backend benchmark script
Simulation experiments.ipynb  Simulation experiments for the manuscript
pyproject.toml                Project environment managed by uv
case_study_for_mirna/         miRNA case study, trained models, and evaluation instructions
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

## Core DiscrimAlign usage

The main function is `discrimalign` from `src.discrimalign`.

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

`stepfunction` maps the iteration number to a step size and is required whenever `max_iter > 0`; `src.optimization` provides `create_powerstep` and `create_constant_step`.

`num_threads=0`, the default, chooses the thread count automatically: all logical cores with the nwgrad backend, and one thread with the Biopython backend, whose threads contend for Python's global interpreter lock and only slow it down. Any other value is used as given. For long sequences such as full-length proteins, the number of physical cores can be faster than all logical cores; pass it explicitly.

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
- Empty sequences and a missing `stepfunction` with `max_iter > 0` raise a `ValueError` before any alignment work.
- The default backend is nwgrad, and `num_threads` defaults to automatic.

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
