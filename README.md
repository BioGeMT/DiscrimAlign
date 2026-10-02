# DiscrimAlign

DiscrimAlign provides a ready-to-run miRNA sequence-pair inference workflow backed by bundled trained models, plus the research code used to train and evaluate those models. The main user-facing path is: provide a CSV of sequence pairs, choose a bundled miRNA model, and receive probabilities plus human-readable alignments.

## Repository structure

```text
src/                         Core DiscrimAlign training and inference code
examples/mirna_pairs.csv       Ready-to-run inference input example
case_study_for_mirna/         Bundled miRNA trained models and evaluation workflows
Simulation experiments.ipynb  Simulation experiments for the manuscript
pyproject.toml                Project environment managed by uv
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

## miRNA Inference Quickstart

Use this workflow when you want predictions from the bundled trained miRNA models without retraining.

### 1. Inspect the Example Input

A ready-to-run CSV is provided at:

```text
examples/mirna_pairs.csv
```

It uses this schema:

```csv
id,sequence_a,sequence_b
example_positive_like,AUGCUA,AUGGUA
example_short,CUGA,CUGU
```

Required columns:

- `sequence_a`: first miRNA/RNA sequence
- `sequence_b`: second miRNA/RNA sequence
- any extra columns, such as `id`, are preserved in the output

### 2. Run a Bundled Model

Two trained miRNA model aliases are available:

- `manakov`: `case_study_for_mirna/trained_models/manakov_best_model.pkl`
- `hejret`: `case_study_for_mirna/trained_models/hejret_best_model.pkl`

Run inference:

```bash
uv run python -m src.infer \
  --model manakov \
  --input examples/mirna_pairs.csv \
  --output predictions_manakov.csv
```

Or use the Hejret-trained model:

```bash
uv run python -m src.infer \
  --model hejret \
  --input examples/mirna_pairs.csv \
  --output predictions_hejret.csv
```

### 3. Read the Output

The output CSV includes:

- `probability`: logistic model probability for the positive class
- `alignment_score`: score assigned by the fitted aligner
- `aligned_sequence_a`, `alignment_marks`, `aligned_sequence_b`: readable alignment
- `operations`: per-position `match`, `mismatch`, or `gap`
- `normalized_sequence_a`, `normalized_sequence_b`: sequences actually scored by the model

By default, `--normalize auto` converts `U`/`T` to match the trained model alphabet. Use `--normalize none` only if you want to disable this behavior.

If your second sequence column contains target/gene sequences before reverse-complementing, pass `--reverse-complement-b` so the second sequence is reverse-complemented before normalization and scoring:

```bash
uv run python -m src.infer \
  --model manakov \
  --input my_pairs.csv \
  --output my_predictions.csv \
  --seq-a-column noncodingRNA \
  --seq-b-column gene \
  --reverse-complement-b
```

### 4. Use Your Own CSV Columns

If your input columns have different names, pass them explicitly:

```bash
uv run python -m src.infer \
  --model manakov \
  --input my_pairs.csv \
  --output my_predictions.csv \
  --seq-a-column mirna \
  --seq-b-column target
```

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

## Training API

Use `discrimalign` directly when you want to fit a new model instead of using the bundled miRNA models.

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
If `stepfunction` is omitted, `discrimalign` uses a conservative default power step with scale `1e-4`.

For inference on new sequence pairs, use `predict_pairs` with the fitted result:

```python
from src import predict_pairs

rows = predict_pairs(
    seqlistA=["AUGCUA"],
    seqlistB=["AUGGUA"],
    model=result,
)

print(rows[0]["probability"])
print(rows[0]["aligned_sequence_a"])
print(rows[0]["alignment_marks"])
print(rows[0]["aligned_sequence_b"])
```

Each inference row is a plain dictionary with the input sequences, normalized sequences, alignment score, logistic probability, aligned strings, match markers, and per-position operations (`match`, `mismatch`, or `gap`). By default, inference normalizes `U`/`T` automatically to match the fitted model alphabet; pass `--normalize none` in the CLI to disable this.

To persist a fitted model and run CSV inference later:

```python
from src import save_model

save_model(result, "model.pkl")
```

Run inference with a custom saved model:

```bash
uv run python -m src.infer --model model.pkl --input examples/mirna_pairs.csv --output predictions.csv
```

Parallel alignment during fitting is chunked when `num_threads > 1`. Each joblib task processes a chunk of sequence pairs rather than a single pair, which reduces scheduler overhead across repeated optimization iterations while preserving alignment order. Thread-based joblib workers are used for the chunked alignment tasks.

## miRNA case study

The miRNA case study, trained models, and instructions for reproducing the manuscript AUPRC metrics are documented in:

```text
case_study_for_mirna/README.md
```
