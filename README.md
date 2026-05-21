# DiscrimAlign

DiscrimAlign is a research codebase for discriminatively learning alignment parameters from labelled pairs of biological sequences. The repository contains the core implementation of the method, the simulation experiments used in the manuscript, and a miRNA case-study workflow.

## Repository structure

```text
src/                          Core DiscrimAlign implementation
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

The miRNA workflow is contained in:

```text
case_study_for_mirna/
```

Dataset access is handled through the `miRBench` package during the run.

Run a two-configuration Hejret calculation:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --final-max-iter 0 --num-threads 1
```

The script

```text
case_study_for_mirna/run_mirna_auprc_table.py
```

contains a convenience workflow for running the miRNA grid on the Hejret and Manakov training families and evaluating each fitted model on `hejret_test`, `manakov_test`, and `manakov_leftout`.

The manuscript-result values in the current draft were generated with explicit case-study CLI runs, including grid-search runs, warm-start continuation, train-only runs with `--skip-evaluation`, and evaluation-only runs with `--max-iters 0`. The exact commands should be recorded with the generated run directories under `results/case_study_for_mirna/`.

Run the convenience workflow from the repository root:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

Case-study outputs are written to:

```text
results/case_study_for_mirna/
```

Each run directory contains grid summaries, metrics, curve points, trajectory files, and plots. Final-refit outputs are written under the corresponding `final_refit/` directory.

### Warm-starting from a saved case-study model

The miRNA case-study CLI can continue fitting from a saved `model.pkl` artifact by passing `--warm-start-model`. This is useful after a grid search has already selected a strong configuration and you want to run additional optimization iterations without restarting from the default initialization.

A saved model artifact is written for every successful grid configuration under:

```text
results/case_study_for_mirna/<run_name>/model_artifacts/<config_name>/model.pkl
```

The selected grid model is also copied to:

```text
results/case_study_for_mirna/<run_name>/best_grid_model/model.pkl
```

Warm-start compatibility is checked against the requested `aligner_mode`, `gap_mode`, and `substitution_mode`. The continuation run should normally use the same configuration family as the saved model.

Example: continue the selected Manakov local-affine-general model for 50 additional iterations:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset manakov \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --aligner-modes local \
  --gap-modes affine \
  --substitution-modes general \
  --stepfunctions constant \
  --step-scales 0.0005 \
  --max-iters 50 \
  --final-max-iter 0 \
  --num-threads 80 \
  --config-workers 1 \
  --warm-start-model results/case_study_for_mirna/<run_name>/best_grid_model/model.pkl \
  --run-tag best_manakov_affine_general_warmstart_plus50_threads80
```

`--max-iters` is the number of additional subgradient iterations to run from the saved model. For example, if the saved model came from a 200-iteration run, `--max-iters 50` performs 50 more iterations from that fitted aligner rather than rerunning 250 iterations from scratch.

The continuation run writes a new run directory, saves a new `model.pkl`, and copies the selected continuation model to its own `best_grid_model/` directory.

### Train-only runs with skipped evaluation

For long continuation runs, use `--skip-evaluation` to fit and save model artifacts without scoring `fit`, `validation`, or evaluation splits and without generating PR/ROC curves. This is useful when the training step is the bottleneck and full Manakov test scoring should be delayed until after a promising continuation model has been saved.

Example: continue the selected Manakov model for 50 additional iterations and save the model without evaluation:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset manakov \
  --aligner-modes local \
  --gap-modes affine \
  --substitution-modes general \
  --stepfunctions constant \
  --step-scales 0.0005 \
  --max-iters 50 \
  --final-max-iter 0 \
  --num-threads 80 \
  --config-workers 1 \
  --warm-start-model results/case_study_for_mirna/<run_name>/best_grid_model/model.pkl \
  --skip-evaluation \
  --run-tag best_manakov_affine_general_warmstart_plus50_train_only
```

When evaluation is skipped, `summary.csv` still records the fitting summary, runtime, final log-likelihood, and model artifact paths. The selected train-only model is copied to `best_grid_model/` using final log-likelihood for ranking. A later run can evaluate the saved `best_grid_model/model.pkl` by using it as `--warm-start-model` with `--max-iters 0` and normal evaluation splits.