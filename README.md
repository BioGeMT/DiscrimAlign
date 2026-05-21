# DiscrimAlign

DiscrimAlign is a research codebase for discriminatively learning alignment parameters from labelled pairs of biological sequences. The repository contains the core implementation of the method, the simulation experiments used in the manuscript, a stable `uv` environment, and a manuscript-aligned miRNA case study with bundled trained models.

## Repository structure

```text
src/                                      Core DiscrimAlign implementation
Simulation experiments.ipynb               Simulation experiments for the manuscript
pyproject.toml                             Project environment managed by uv
case_study_for_mirna/                      miRNA case-study workflow
case_study_for_mirna/trained_models/       Selected miRNA models and evaluation instructions
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

The miRNA case study is contained in:

```text
case_study_for_mirna/
```

The repository includes selected trained miRNA models used in the manuscript under:

```text
case_study_for_mirna/trained_models/
```

These bundled models allow readers to reproduce the reported AUPRC metrics on the manuscript evaluation sets without rerunning model fitting.

### Reproduce manuscript metrics from bundled models

Use the bundled `model.pkl` artifacts with `--warm-start-model` and `--max-iters 0`. This loads the trained model and evaluates it on the requested datasets.

Hejret-trained selected model:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset hejret \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --aligner-modes local \
  --gap-modes affine \
  --substitution-modes general \
  --stepfunctions constant \
  --step-scales 0.0005 \
  --max-iters 0 \
  --final-max-iter 0 \
  --num-threads 8 \
  --config-workers 1 \
  --warm-start-model case_study_for_mirna/trained_models/hejret_selected_model/model.pkl \
  --run-tag manuscript_hejret_model_eval
```

Manakov-trained selected model:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset manakov \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --aligner-modes local \
  --gap-modes affine \
  --substitution-modes general \
  --stepfunctions constant \
  --step-scales 0.0005 \
  --max-iters 0 \
  --final-max-iter 0 \
  --num-threads 8 \
  --config-workers 1 \
  --warm-start-model case_study_for_mirna/trained_models/manakov_selected_model/model.pkl \
  --run-tag manuscript_manakov_model_eval
```

Key output files are written under `results/case_study_for_mirna/<dataset>_<run-tag>/`:

```text
summary.csv
metrics.csv
pr_points.csv
roc_points.csv
best_grid_model/selected_summary.json
```

### Evaluate user-provided evaluation sets

In addition to miRBench aliases passed through `--eval-splits`, users can provide their own CSV or TSV evaluation files with `--eval-files`. Each file must contain:

```text
noncodingRNA,gene,label
```

The `gene` column should contain the target sequence before reverse complementing. The script reverse-complements it internally to match the manuscript workflow.

Example:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset manakov \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --eval-files external_set=path/to/external_set.tsv \
  --aligner-modes local \
  --gap-modes affine \
  --substitution-modes general \
  --stepfunctions constant \
  --step-scales 0.0005 \
  --max-iters 0 \
  --final-max-iter 0 \
  --num-threads 8 \
  --config-workers 1 \
  --warm-start-model case_study_for_mirna/trained_models/manakov_selected_model/model.pkl \
  --run-tag manuscript_manakov_model_external_eval
```

The evaluation commands write one metrics table per run and store precision-recall and ROC curve points for each evaluated split.
