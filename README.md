# EstimAlign

EstimAlign is a research codebase for estimating alignment parameters from labelled pairs of biological sequences. The main components of the project are the implementation in `src/` and the simulation experiments notebook.

The optional miRNA case study is included only as supplementary manuscript-review material. It is not required for the core method, the simulation experiments, or the main use of EstimAlign.

## Repository contents

```text
src/                         Core EstimAlign implementation
Simulation experiments.ipynb  Simulation experiments for the manuscript
pyproject.toml                Reproducible uv environment
case_study_for_mirna/         Optional miRNA review workflow
```

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) to create a reproducible Python environment.

Install the core environment:

```bash
uv sync
```

This is sufficient for the main `src/` scripts and simulation experiments.

To run the optional miRNA case-study workflow, install the extra dependencies:

```bash
uv sync --extra case-study
```

The `case-study` extra installs `miRBench`, which is used only to download or reuse cached miRNA benchmark datasets for manuscript-review checks.

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

The returned object contains the fitted aligner, learned parameters, intercept, final log-likelihood, and optimization trajectories.

## Simulation experiments

The simulation notebook is the main reproducibility material for the manuscript experiments:

```text
Simulation experiments.ipynb
```

Open it with a notebook interface configured to use the `uv` environment. For example:

```bash
uv run jupyter lab "Simulation experiments.ipynb"
```

If Jupyter is not installed in the environment, install it in `.venv` or open the notebook from an editor that can use the project environment.

## Optional miRNA case study

The folder `case_study_for_mirna/` is provided so reviewers or readers can rerun selected miRNA case-study checks. The scripts obtain datasets through the `miRBench` package and reuse cached files when available.

Install the optional dependency group before running these scripts:

```bash
uv sync --extra case-study
```

List available dataset aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download or reuse the default Hejret data:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Run a small smoke test:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

Case-study outputs are written to:

```text
results/case_study_for_mirna/
```

## Data policy

Raw miRBench datasets are not stored as primary repository artifacts. They are requested through `miRBench` only when the optional case-study scripts are run.
