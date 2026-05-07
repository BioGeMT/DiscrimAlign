# miRNA case study for EstimAlign

This folder contains the miRNA case-study workflow for the `uv_case_study` branch.

## Files

- `import_mirbench_datasets.py`: downloads/caches benchmark datasets through the `miRBench` Python package and writes them into the repository's `data/raw/` layout.
- `case_study_mirna.py`: runs an EstimAlign grid on paired train/test datasets or on a single split.

## Setup

From the repository root:

```bash
uv sync
```

## Dataset download

List dataset aliases and the datasets exposed by `miRBench`:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download the default Hejret group:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Download the miRNA table group:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups table
```

Download specific aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups "" --datasets hejret_train,manakov_leftout,klimentova_test
```

## Pipeline examples

Small smoke test:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

Fuller default Hejret run:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --num-threads 1
```

Direct single-split run:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset-split manakov_leftout --limit-configs 10 --num-threads 1
```

Outputs are written to `results/case_study_for_mirna/` by default.
