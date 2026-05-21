# miRNA case study

This directory contains the manuscript-aligned miRNA case study for DiscrimAlign. It includes trained models and scripts for reporting AUPRC metrics on the miRBench evaluation splits and on user-provided evaluation files.

The selected trained models used in the manuscript are stored under:

```text
case_study_for_mirna/trained_models/
```

These models are included for reproducibility and for reporting AUPRC metrics on the manuscript evaluation sets.

## Files

- `case_study_mirna.py`: evaluates trained DiscrimAlign models and writes metrics, curve points, summaries, and model metadata.
- `import_mirbench_datasets.py`: maps dataset aliases to the corresponding `miRBench` datasets.
- `scoring.py`, `modeling.py`, and `outputs.py`: support model loading, scoring, output writing, and plotting for the case-study workflow.
- `trained_models/`: selected trained models used for manuscript evaluation.

## Environment

Create the project environment from the repository root:

```bash
uv sync
```

All commands below should be run from the repository root with `uv run`.

## Trained models

The trained models are stored as:

```text
case_study_for_mirna/trained_models/
  hejret_best_model.pkl
  manakov_best_model.pkl
```

If the model files are stored with Git LFS, make sure Git LFS files have been pulled before running evaluation.

## Reproduce manuscript metrics

Use the trained model artifacts with `--warm-start-model` and `--max-iters 0`. This loads the model and evaluates it on the requested datasets.

### Hejret-trained selected model

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
  --warm-start-model case_study_for_mirna/trained_models/hejret_best_model.pkl \
  --run-tag manuscript_hejret_model_eval
```

### Manakov-trained selected model

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
  --warm-start-model case_study_for_mirna/trained_models/manakov_best_model.pkl \
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

The `summary.csv` file contains the AUPRC and ROC-AUC values for each evaluated split. The `pr_points.csv` and `roc_points.csv` files contain the precision-recall and ROC curve points used to reproduce plots.

## Evaluate user-provided evaluation sets

In addition to miRBench aliases passed through `--eval-splits`, users can provide their own CSV or TSV evaluation files with `--eval-files`. Each file must contain:

```text
noncodingRNA,gene,label
```

- `noncodingRNA`: miRNA sequence
- `gene`: target sequence before reverse complementing
- `label`: binary label, with 1 for confirmed interaction and 0 for negative pair

The script reverse-complements the `gene` column internally before alignment, matching the manuscript workflow.

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
  --warm-start-model case_study_for_mirna/trained_models/manakov_best_model.pkl \
  --run-tag manuscript_manakov_model_external_eval
```

The metrics for the user-provided set are added to the same output files as the miRBench evaluation splits.
