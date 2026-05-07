# Optional miRNA case study

This folder contains optional manuscript-review material for EstimAlign. The main project is the core method implementation in `src/` and the simulation experiments notebook. The scripts here are provided only so reviewers or readers can check selected miRNA case-study results when needed.

The case study uses the `miRBench` package to obtain datasets. Dataset files are downloaded or reused from cache only when these scripts are run; they are not required for the main EstimAlign code or simulations.

## Files

- `import_mirbench_datasets.py`: resolves EstimAlign-friendly dataset aliases, asks `miRBench` for the corresponding dataset, and writes/reuses local files under `data/raw/`.
- `case_study_mirna.py`: runs a compact EstimAlign grid on paired train/test datasets or a direct single-split check.

## Install optional dependencies

From the repository root:

```bash
uv sync --extra case-study
```

The base command `uv sync` is enough for the main `src/` code and simulations. The extra is needed only for the miRNA review workflow.

## Dataset download and cache

List dataset aliases and the datasets exposed by `miRBench`:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download or reuse the default Hejret group:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Download the datasets used for manuscript table checks:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups table
```

Download specific aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups "" --datasets hejret_train,manakov_leftout,klimentova_test
```

## Pipeline examples

Small smoke test for checking that the optional workflow runs:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

A fuller default Hejret review run:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --num-threads 1
```

Direct single-split check:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset-split manakov_leftout --limit-configs 10 --num-threads 1
```

Outputs are written to `results/case_study_for_mirna/` by default.
