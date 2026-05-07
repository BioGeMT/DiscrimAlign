# EstimAlign

Finding substitution weights and gap penalties from pairs of labeled biological sequences.

## Branch purpose: `uv_case_study`

This branch keeps the `src/` implementation, simulation notebook, and baseline README from `devel`, and adds a reproducible `uv` environment for the miRNA case study.

The new case-study code lives in:

```text
case_study_for_mirna/
```

It uses the `miRBench` Python package to download and load the miRNA benchmark datasets instead of storing raw datasets in the repository.

## Environment setup with `uv`

Install `uv` if needed, then create the environment from the repository root:

```bash
uv sync
```

Run commands inside the environment with:

```bash
uv run python --version
```

## Download miRBench datasets

List supported datasets and aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download the default Hejret train/test datasets into `data/raw/`:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Download the datasets used for the miRNA table:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups table
```

Download specific aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups "" --datasets hejret_train,manakov_leftout,klimentova_test
```

Supported aliases are:

- `hejret_train`
- `hejret_test`
- `manakov_train`
- `manakov_test`
- `manakov_leftout`
- `klimentova_test`

## Run the miRNA case-study pipeline

Run a compact default grid on the Hejret train/test split:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --num-threads 1
```

Run only a small smoke-test grid:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

Run a single split directly:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset-split manakov_leftout --limit-configs 10 --num-threads 1
```

Pipeline summaries are written to:

```text
results/case_study_for_mirna/
```

## Repository layout

- `src/`: core EstimAlign scripts from `devel`
- `Simulation experiments.ipynb`: notebook from `devel`
- `case_study_for_mirna/`: miRNA case-study workflow using `miRBench`
- `pyproject.toml`: `uv` project configuration
