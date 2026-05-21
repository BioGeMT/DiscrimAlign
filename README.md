# DiscrimAlign

DiscrimAlign is a research codebase for discriminatively learning alignment parameters from labelled pairs of biological sequences. The repository contains the core implementation of the method, the simulation experiments used in the manuscript, a stable `uv` environment, and a manuscript-aligned miRNA case study with bundled trained models.

## Repository structure

```text
src/                                      Core DiscrimAlign implementation
Simulation experiments.ipynb               Simulation experiments for the manuscript
pyproject.toml                             Project environment managed by uv
case_study_for_mirna/                      miRNA case-study workflow, trained models, and evaluation instructions
```

## Requirements

- Python `>=3.10,<3.13`
- `uv` for environment management
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
    num_threads=1,
)

print(result["final_loglik"])
print(result["alpha"])
```

The returned object contains the fitted aligner, learned alignment parameters, intercept, final log-likelihood, and optimization trajectories.

Parallel alignment during fitting is chunked when `num_threads > 1`. Each joblib task processes a chunk of sequence pairs rather than a single pair, which reduces scheduler overhead across repeated optimization iterations while preserving alignment order. Thread-based joblib workers are used for the chunked alignment tasks.

## miRNA case study

The manuscript-aligned miRNA case study, bundled trained models, and instructions for reproducing the reported evaluation metrics are documented in:

```text
case_study_for_mirna/README.md
```
