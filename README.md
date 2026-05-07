# EstimAlign

EstimAlign is a research codebase for estimating alignment parameters from labelled pairs of biological sequences. The repository contains the core implementation of the method, the simulation experiments used in the manuscript, and a miRNA case-study workflow.

## Repository structure

```text
src/                          Core EstimAlign implementation
Simulation experiments.ipynb   Simulation experiments for the manuscript
pyproject.toml                 Project environment managed by uv
case_study_for_mirna/          miRNA case-study workflow and manuscript-result runs
```

The miRNA case-study workflow uses `miRBench` for dataset access and writes its outputs to `results/case_study_for_mirna/`.

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

## miRNA case study

The miRNA workflow is contained in:

```text
case_study_for_mirna/
```

Dataset access is handled through the `miRBench` package during the run.

Run a two-configuration example on the Hejret data:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

The script

```text
case_study_for_mirna/run_mirna_auprc_table.py
```

contains the miRNA runs used to reproduce the corresponding results reported in the manuscript. It trains once on the Hejret family and once on the Manakov family, evaluating each fitted model on `hejret_test`, `manakov_test`, and `manakov_leftout`.

Run the manuscript-result workflow from the repository root:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

Case-study outputs are written to:

```text
results/case_study_for_mirna/
```
