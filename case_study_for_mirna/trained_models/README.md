# Trained miRNA models

This directory is reserved for the trained DiscrimAlign miRNA models used in the manuscript. The repository should include the selected fitted models so that readers can reproduce the reported evaluation metrics without rerunning the full grid search or long continuation fits.

## Expected layout

```text
case_study_for_mirna/trained_models/
  hejret_selected_model/
    model.pkl
    model_parameters.json
    selected_summary.json
  manakov_selected_model/
    model.pkl
    model_parameters.json
    selected_summary.json
```

The model directories should be copied from the corresponding `best_grid_model/` directories produced by `case_study_mirna.py`. For example:

```bash
mkdir -p case_study_for_mirna/trained_models
cp -r results/case_study_for_mirna/<hejret-run>/best_grid_model \
  case_study_for_mirna/trained_models/hejret_selected_model
cp -r results/case_study_for_mirna/<manakov-run>/best_grid_model \
  case_study_for_mirna/trained_models/manakov_selected_model
```

If the `model.pkl` files are large, store them with Git LFS rather than regular Git blobs.

## Reproducing manuscript metrics from bundled models

The case-study CLI can evaluate a saved model without additional fitting by combining `--warm-start-model` with `--max-iters 0`. The evaluation sets may come from miRBench aliases via `--eval-splits` or from user-provided CSV/TSV files via `--eval-files`.

### Evaluate the bundled Hejret-trained model on miRBench splits

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

### Evaluate the bundled Manakov-trained model on miRBench splits

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

Outputs are written to `results/case_study_for_mirna/<dataset>_<run-tag>/`. The key files are:

```text
summary.csv
metrics.csv
pr_points.csv
roc_points.csv
best_grid_model/selected_summary.json
```

## Evaluating user-provided datasets

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
  --warm-start-model case_study_for_mirna/trained_models/manakov_selected_model/model.pkl \
  --run-tag manuscript_manakov_model_external_eval
```

The script reverse-complements the `gene` column internally before alignment, matching the manuscript workflow.
