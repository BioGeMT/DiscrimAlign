# EstimAlign

EstimAlign is a research codebase for estimating alignment parameters from labelled pairs of biological sequences. The central components of the repository are the mathematical and computational implementation in `src/` and the simulation experiments notebook.

## Repository structure

```text
src/                         Core EstimAlign implementation
Simulation experiments.ipynb  Simulation experiments for the manuscript
pyproject.toml                Reproducible project environment managed by uv
case_study_for_mirna/         miRNA case-study workflow
```

## Requirements

- Python `>=3.10,<3.13`
- `uv` for environment management
- Jupyter or VS Code notebook support for running `Simulation experiments.ipynb`

The repository provides one main project environment through `uv`. The optional `case-study` dependency group adds the package needed by the miRNA case-study scripts to obtain datasets through `miRBench`. The mathematical implementation in `src/`, the simulation notebook, and the case-study scripts are otherwise intended to run in the same project environment.

## Installing `uv`

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal, then verify the installation:

```powershell
uv --version
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal, then verify the installation:

```bash
uv --version
```

## Creating the project environment

From the repository root, install the standard project dependencies:

```bash
uv sync
```

This creates a local `.venv/` environment for the main implementation and simulation experiments.

For the full reproducible research environment, including the miRNA case-study dataset interface, use:

```bash
uv sync --extra case-study
```

The difference is that `--extra case-study` also installs `miRBench`, which is used by `case_study_for_mirna/` to download or reuse cached miRNA benchmark datasets.

To verify that the environment is available:

```bash
uv run python --version
```

## Installing Jupyter

Jupyter is needed to run the simulation notebook through JupyterLab. It is not installed by `uv sync` unless it is added explicitly.

To install JupyterLab into the project environment:

```bash
uv add jupyterlab ipykernel
```

Then open the notebook with:

```bash
uv run jupyter lab "Simulation experiments.ipynb"
```

Alternatively, to use JupyterLab temporarily without modifying `pyproject.toml`:

```bash
uv run --with jupyterlab jupyter lab "Simulation experiments.ipynb"
```

### Using VS Code

If using VS Code, install the Python and Jupyter extensions. Then:

1. Run `uv sync` or `uv sync --extra case-study` from the repository root.
2. Open `Simulation experiments.ipynb`.
3. Select the kernel associated with the local `.venv/` environment.

If the `.venv` kernel is not listed, run:

```bash
uv add ipykernel
```

Then reopen the notebook or refresh the kernel list in VS Code.

## Core EstimAlign usage

The main function is `estimalign` from `src.estimalign`.

```python
from src.estimalign import estimalign

seqlistA = ["AUGCUA", "CUGA"]
seqlistB = ["AUGGUA", "CUGU"]
labels = [1, 0]

result = estimalign(
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

## Simulation experiments

The notebook

```text
Simulation experiments.ipynb
```

contains the simulation experiments associated with the manuscript. It should be treated as the primary reproducibility material alongside the implementation in `src/`.

After installing Jupyter, run:

```bash
uv run jupyter lab "Simulation experiments.ipynb"
```

or open the notebook in VS Code using the `.venv/` kernel.

## miRNA case study

The folder

```text
case_study_for_mirna/
```

contains scripts for reproducing selected miRNA case-study calculations. This workflow uses the `miRBench` package to obtain datasets and reuses cached files when available.

Install the project environment with the case-study dependency group:

```bash
uv sync --extra case-study
```

List the available dataset aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download or reuse the default Hejret data:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Run a small verification example with two grid configurations:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

Case-study outputs are written to:

```text
results/case_study_for_mirna/
```

## Data policy

Raw miRBench datasets are not stored as primary repository artifacts. They are requested through `miRBench` only when the case-study scripts are run.
