# EstimAlign

EstimAlign estimates substitution weights and gap penalties from pairs of labelled biological sequences. The main project deliverables are the core implementation in `src/` and the simulation experiments notebook.

The repository is organized as a reproducible research codebase for the manuscript. The miRNA case study is included as optional review material: it is provided so reviewers or readers can download the relevant miRBench datasets and rerun selected checks, but it is not the main product of the repository.

## Main project contents

- `src/`: core EstimAlign implementation and optimization scripts.
- `Simulation experiments.ipynb`: simulation experiments used to evaluate the method.
- `pyproject.toml`: reproducible `uv` environment for running the core project.
- `case_study_for_mirna/`: optional manuscript-review case study using datasets obtained through the `miRBench` package.

## Environment setup

The project uses [`uv`](https://docs.astral.sh/uv/) for reproducible environment management.

For the core EstimAlign code and simulations:

```bash
uv sync
```

Then run Python commands through the environment:

```bash
uv run python --version
```

To include the optional miRNA manuscript-review case study dependencies:

```bash
uv sync --extra case-study
```

This installs `miRBench` only when the case-study workflow is needed.

## Core usage

The primary entry point is `src.estimalign.estimalign`.

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

The returned dictionary contains the learned alignment parameters, fitted aligner, intercept, optimization trajectory, and final objective value.

## Simulation experiments

The simulation notebook is part of the main project and should be treated as the primary reproducibility material alongside the `src/` scripts.

Start Jupyter from the `uv` environment and open the notebook:

```bash
uv run jupyter lab "Simulation experiments.ipynb"
```

If Jupyter is not already installed in your environment, install it in the active `uv` environment or run the notebook through your preferred editor configured to use `.venv`.

## Optional miRNA case study for manuscript review

The folder `case_study_for_mirna/` is optional. It exists to support manuscript review and result checking for the miRNA case study. It downloads or reuses cached miRBench datasets only when the user explicitly runs the case-study scripts.

Install the optional dependency first:

```bash
uv sync --extra case-study
```

List the available dataset aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download the default cached Hejret train/test data into `data/raw/`:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Run a small smoke test of the manuscript-review pipeline:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

The case-study outputs are written under:

```text
results/case_study_for_mirna/
```

## Notes on data

Raw miRBench datasets are not the main repository artifact. The optional case-study scripts request the datasets through the `miRBench` package and then use cached or downloaded files when available.
