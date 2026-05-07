# miRNA case study

This folder contains the miRNA case-study workflow for EstimAlign. The scripts use the same `uv` project environment as the main repository and obtain datasets through the `miRBench` package.

Dataset files are downloaded or reused from cache when the scripts are run. Raw miRBench datasets are not stored as primary repository artifacts.

## Files

- `import_mirbench_datasets.py`: resolves EstimAlign dataset aliases, obtains the corresponding datasets through `miRBench`, and writes or reuses local files under `data/raw/`.
- `case_study_mirna.py`: runs an EstimAlign grid on paired train/test datasets or on a single dataset split.

## Environment

From the repository root:

```bash
uv sync
```

This installs the dependencies used by the EstimAlign implementation, the simulation notebook, and the miRNA case-study workflow.

## Datasets

List dataset aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --list
```

Download or reuse the default Hejret group:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py
```

Download the datasets used for miRNA table calculations:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups table
```

Download specific aliases:

```bash
uv run python case_study_for_mirna/import_mirbench_datasets.py --groups "" --datasets hejret_train,manakov_leftout,klimentova_test
```

## Pipeline examples

Two-configuration Hejret example:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --num-threads 1
```

Default Hejret run:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --num-threads 1
```

Single-split Manakov run:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset-split manakov_leftout --limit-configs 10 --num-threads 1
```

Outputs are written to:

```text
results/case_study_for_mirna/
```
