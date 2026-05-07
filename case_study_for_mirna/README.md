# miRNA case study

This directory contains the miRNA case-study workflow for EstimAlign. The scripts use the same `uv` project environment as the main repository and access datasets through the `miRBench` package during execution.

## Files

- `import_mirbench_datasets.py`: maps EstimAlign dataset aliases to the corresponding `miRBench` datasets.
- `case_study_mirna.py`: runs an EstimAlign grid on paired train/test datasets or on a single dataset split.
- `run_mirna_auprc_table.py`: contains the miRNA runs used to reproduce the corresponding manuscript results.

## Environment

Create the project environment from the repository root:

```bash
uv sync
```

The same environment is used for the EstimAlign implementation, simulation notebook, and miRNA case-study workflow.

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

## Manuscript-result runs

The script

```text
case_study_for_mirna/run_mirna_auprc_table.py
```

contains the runs used to reproduce the miRNA results included in the manuscript. It trains once on the Hejret family and once on the Manakov family, evaluating each fitted model on `hejret_test`, `manakov_test`, and `manakov_leftout`.

Run the manuscript-result workflow from the repository root:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

Outputs are written to:

```text
results/case_study_for_mirna/
```
