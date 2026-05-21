# Trained miRNA models

This directory contains the selected trained DiscrimAlign miRNA models used in the manuscript. These models are included for reproducibility and for reporting AUPRC metrics on the manuscript evaluation sets.

## Model files

```text
case_study_for_mirna/trained_models/
  hejret_best_model.pkl
  manakov_best_model.pkl
```

If the `model.pkl` files are stored with Git LFS, make sure Git LFS files have been pulled before running evaluation.

## Reproduce manuscript metrics

The case-study CLI can evaluate a saved model without additional fitting by combining `--warm-start-model` with `--max-iters 0`. The evaluation sets may come from miRBench aliases via `--eval-splits` or from user-provided CSV/TSV files via `--eval-files`.

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

Outputs are written to `results/case_study_for_mirna/<dataset>_<run-tag>/`. The key files are:

```text
summary.csv
metrics.csv
pr_points.csv
roc_points.csv
best_grid_model/selected_summary.json
```

## Evaluate user-provided datasets

User-provided evaluation sets must be CSV or TSV files with these columns:

```text
noncodingRNA,gene,label
```

- `noncodingRNA`: miRNA sequence
- `gene`: target sequence, before reverse complementing
- `label`: binary label, with 1 for confirmed interaction and 0 for negative pair

Use `--eval-files` to add one or more named evaluation files:

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

The script reverse-complements the `gene` column internally before alignment, matching the manuscript workflow.
