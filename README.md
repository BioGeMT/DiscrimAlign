# EstimAlign

EstimAlign is a Python toolkit for estimating sequence-alignment parameters from labeled pairs of biological sequences. It learns substitution weights and gap penalties from data and reports optimization summaries, parameter-recovery tables, terminal logs, and diagnostic figures.

This repository now includes a `uv`-managed Python environment and a command-line workflow converted from the non-protein simulation sections of the original simulation notebook.

The current converted workflow focuses on miRNA simulation experiments as a validation example. EstimAlign itself is not limited to miRNA data.

## What the converted workflow does

The command-line workflow runs these notebook-derived simulation experiments:

1. Simple alignment model with local alignment, match/mismatch scores, and affine gap penalties.
2. Step-length experiment for comparing optimization behavior across constant step sizes.
3. Replicate experiment for assessing run-to-run variability.
4. General substitution matrix experiment with affine gap penalties.

The workflow writes:

- a JSON summary,
- TSV tables for downstream analysis,
- parameter truth-vs-estimate reports,
- a terminal-output log file,
- optional PNG figures.

## Install uv

This project uses `uv` for Python environment and dependency management.

### Windows PowerShell

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then check:

```powershell
uv --version
```

### Linux / macOS shell

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, or follow the installer's shell-profile instructions. Then check:

```bash
uv --version
```

## Setup the project

Clone the repository and enter the project directory.

### Windows PowerShell

```powershell
git clone <repository-url>
cd EstimAlign
uv sync
```

### Linux / macOS shell

```bash
git clone <repository-url>
cd EstimAlign
uv sync
```

Check that the command-line entrypoint is available:

```bash
uv run estimalign --help
```

## Run the simulation experiments

### Quick smoke test without plots

Use this first to confirm that the environment and workflow are working.

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2 --no-plots
```

The same command works in Windows PowerShell.

### Quick smoke test with plots

Run the smoke test without `--no-plots`:

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2
```

### Full default run

```bash
uv run estimalign run-simulation-experiments
```

The full default run may take longer because it uses more records, more iterations, and plot generation.

### Detailed optimizer logs

Runs write terminal output to `outputs/simulation_experiments/run.log`. Runs are quiet unless you add `--verbose`:

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2 --no-plots --verbose
```

## Command options

```text
--output-dir             Directory for generated outputs.
--dataset-index          Dataset index from miRBench list_datasets().
--split                  miRBench split to use. Default: train.
--random-seed            Random seed for reproducibility.
--max-records            Optional row limit for quick tests.
--num-threads            Number of EstimAlign worker threads.
--simple-max-iter        Iterations for simple and general simulation experiments.
--step-max-iter          Iterations for the step-length experiment.
--replicate-max-iter     Iterations for each replicate experiment.
--replicate-count        Number of replicate runs.
--no-plots               Disable PNG figure generation.
--verbose                Print detailed optimizer progress to the run log.
```

## Outputs

By default, generated outputs are written to:

```text
outputs/simulation_experiments/
```

### Run log

```text
outputs/simulation_experiments/run.log
```

This file captures terminal output from the run, including miRBench messages and detailed optimizer progress when `--verbose` is used.

### JSON summary

```text
outputs/simulation_experiments/simulation_summary.json
```

This file contains the run configuration, dataset summary, per-experiment results, and parameter-comparison sections.

### Parameter recovery reports

These files compare the simulated truth against the fitted EstimAlign parameters:

```text
outputs/simulation_experiments/simple_parameter_comparison.tsv
outputs/simulation_experiments/general_parameter_comparison.tsv
```

Each table contains:

```text
parameter    true    estimated    absolute_error
```

These reports are useful for checking whether EstimAlign recovers the expected alignment parameters from the simulated labels.

### Experiment result tables

```text
outputs/simulation_experiments/step_length_results.tsv
outputs/simulation_experiments/replicate_results.tsv
outputs/simulation_experiments/general_substitution_matrix_comparison.tsv
```

These TSV files are designed to be easy to inspect in spreadsheet software or load into R, Python, or other analysis tools.

## How to interpret outputs

Start with the parameter recovery reports:

```text
simple_parameter_comparison.tsv
general_parameter_comparison.tsv
```

The `true` column is the parameter value used to simulate labels. The `estimated` column is the value learned by EstimAlign. The `absolute_error` column is the recovery error. Smaller absolute error means the fitted model is closer to the simulated truth.

Use `step_length_results.tsv` to compare optimizer step sizes. Each row is one constant step length. Useful columns are `final_loglik`, `max_loglik`, and the estimated alignment parameters.

Use `replicate_results.tsv` to inspect run-to-run variability. Each row is one replicate with newly sampled labels from the same simulated probability model.

Use `general_substitution_matrix_comparison.tsv` for the general substitution matrix experiment. Each row is one matrix entry. The `char1` and `char2` columns identify the substitution, while `true`, `estimated`, and `absolute_error` compare the simulated and fitted substitution score.

Use `simulation_summary.json` when you need the complete machine-readable record of a run, including configuration, summary metrics, and all nested experiment outputs.

Use `run.log` when you need to inspect terminal messages or detailed optimizer progress.

### Figures

When plots are enabled, figures are written to:

```text
outputs/simulation_experiments/figures/
```

Expected figure files include:

```text
simple_scores_hist.png
simple_logit_scores_hist.png
simple_scores_vs_labels.png
simple_logit_scores_vs_labels.png
simple_optimization_trajectory.png
step_length_loglikelihoods.png
step_length_loglikelihoods_zoom.png
replicate_loglikelihoods.png
general_scores_hist.png
general_logit_scores_hist.png
general_optimization_trajectory.png
general_substitution_comparison.png
```

## Generated files and git

Generated outputs are intentionally ignored by git:

```text
outputs/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Do not commit generated JSON, TSV, PNG, log, cache, or notebook checkpoint files.

## Code layout

Package source lives under:

```text
src/estimalign/
```

Important workflow files:

```text
src/estimalign/cli.py                         Command-line interface.
src/estimalign/simulation_experiments.py      Main orchestration for the simulation workflow.
src/estimalign/simulation_config.py           Dataclasses for run configuration and simulation truths.
src/estimalign/simulation_dataset.py          miRBench dataset loading.
src/estimalign/simulation_model_experiments.py Experiment-specific simulation steps.
src/estimalign/simulation_metrics.py          Scoring, likelihood, and parameter-comparison helpers.
src/estimalign/simulation_outputs.py          JSON and TSV writers.
src/estimalign/simulation_plots.py            Figure generation helpers.
```

The project is packaged through `pyproject.toml` and locked with `uv.lock`.

## Current scope

The current command-line workflow covers the non-protein simulation/validation sections of the original simulation notebook.

The protein experiments are not included in the main command-line workflow yet. They should be converted separately with explicit, configurable input paths before being added to the package.

## Suggested next improvements

- Convert protein experiments into separate configurable commands.
- Add small example output snippets for documentation.
