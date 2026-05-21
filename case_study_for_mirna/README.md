# miRNA case study

This directory contains the miRNA case-study workflow for DiscrimAlign. The scripts use the same `uv` project environment as the main repository and access datasets through the `miRBench` package during execution.

## Files

- `import_mirbench_datasets.py`: maps DiscrimAlign dataset aliases to the corresponding `miRBench` datasets.
- `case_study_mirna.py`: runs the DiscrimAlign grid, evaluates fitted models, writes metrics, trajectories, curve points, and plots, and performs the final refit stage.
- `run_mirna_auprc_table.py`: provides a convenience workflow for running the miRNA grid on the Hejret and Manakov training families.
- `scoring.py`, `modeling.py`, and `outputs.py`: support scoring, model fitting, output writing, and plotting for the case-study workflow.

## Environment

Create the project environment from the repository root:

```bash
uv sync
```

The same environment is used for the DiscrimAlign implementation, simulation notebook, and miRNA case-study workflow.

## Pipeline examples

Two-configuration Hejret calculation:

```bash
uv run python case_study_for_mirna/case_study_mirna.py --dataset hejret --limit-configs 2 --max-iters 5 --final-max-iter 0 --num-threads 1
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

provides a convenience workflow for running the miRNA grid on the Hejret and Manakov training families. It evaluates each fitted model on `hejret_test`, `manakov_test`, and `manakov_leftout`.

The manuscript-result values in the current draft were generated with explicit case-study CLI runs, including grid-search runs, warm-start continuation, train-only runs with `--skip-evaluation`, and evaluation-only runs with `--max-iters 0`. The exact commands should be recorded with the generated run directories under `results/case_study_for_mirna/`.

Run the convenience workflow from the repository root:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

Outputs are written to:

```text
results/case_study_for_mirna/
```

Each run directory contains grid summaries, metrics, curve points, trajectory files, model artifacts, and plots. Final-refit outputs are written under the corresponding `final_refit/` directory.

## Warm-starting and train-only continuation

`case_study_mirna.py` supports continuation from a saved `model.pkl` artifact through `--warm-start-model`. This is useful after a grid search has already selected a strong configuration and additional optimization iterations should start from that fitted model rather than from the default initialization.

For long continuation runs, add `--skip-evaluation` to fit and save model artifacts without scoring `fit`, `validation`, or held-out evaluation splits. A later run can evaluate the saved model by using it as `--warm-start-model` with `--max-iters 0` and normal evaluation splits.