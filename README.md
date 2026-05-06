# EstimAlign

EstimAlign is a Python toolkit for estimating sequence-alignment parameters from labeled pairs of biological sequences. It learns substitution weights and gap penalties from data and reports optimization summaries, parameter-recovery tables, terminal logs, and diagnostic figures.

This repository includes a `uv`-managed Python environment and a command-line workflow for non-protein simulated-interaction validation experiments. The workflow uses miRBench RNA sequence pairs as realistic sequence inputs and simulates labels from known alignment parameters so that parameter recovery can be evaluated directly.

EstimAlign itself is not limited to miRNA data. The RNA simulation workflow is the included validation workflow for this package.

## Simulation workflow

The command-line workflow runs:

1. A simple alignment model with local alignment, match/mismatch scores, and affine gap penalties.
2. A step-length experiment for comparing optimization behavior across constant step sizes.
3. A simple-model replicate experiment for assessing run-to-run variability.
4. A general asymmetric substitution matrix experiment with affine gap penalties.
5. A general/asymmetric replicate experiment matching the robustness check used in the manuscript simulations.

Each run writes a self-contained output directory containing a JSON summary, TSV tables, parameter-recovery reports, a terminal-output log file, an output-specific README, and optional publication-oriented PNG figures.

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

## Setup

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

### Short verification run

Use a small number of records and iterations to verify that the workflow and environment are functioning:

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --general-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2 --no-plots
```

The same command works in Windows PowerShell.

### Short verification run with figures

Run the same reduced configuration without `--no-plots`:

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --general-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2
```

### Full run

```bash
uv run estimalign run-simulation-experiments
```

The full default run uses more records, more iterations, both replicate experiments, and figure generation.

### Detailed optimizer logs

Runs write terminal output to `run.log` inside the run directory. Runs are quiet unless `--verbose` is supplied:

```bash
uv run estimalign run-simulation-experiments --max-records 50 --simple-max-iter 2 --general-max-iter 2 --step-max-iter 2 --replicate-max-iter 2 --replicate-count 2 --no-plots --verbose
```

## Command options

```text
--output-dir             Root directory for generated run folders.
--dataset-index          Dataset index from miRBench list_datasets().
--split                  miRBench split to use. Default: train.
--random-seed            Random seed for reproducibility.
--max-records            Optional row limit for reduced runs.
--num-threads            Number of EstimAlign worker threads.
--simple-max-iter        Iterations for the simple simulation experiment. Default: 50.
--general-max-iter       Iterations for the general matrix experiment. Default: 200.
--step-max-iter          Iterations for the step-length experiment. Default: 10.
--replicate-max-iter     Iterations for each replicate experiment. Default: 5.
--replicate-count        Number of replicate runs. Default: 20.
--simple-step-length     Constant step length for the simple model. Default: 2e-5.
--general-step-length    Constant step length for the general matrix model. Default: 4e-5.
--alpha-mode             Label-simulation alpha mode: fixed or negative-median. Default: negative-median.
--no-plots               Disable PNG figure generation.
--verbose                Print detailed optimizer progress to the run log.
```

`--alpha-mode negative-median` sets the simulation intercept to the negative median alignment score, which produces a roughly balanced simulated label distribution. `--alpha-mode fixed` uses the historical fixed notebook values for alpha.

## Output layout

By default, outputs are organized under:

```text
outputs/simulation_experiments/
```

Each command creates a new timestamped run directory:

```text
outputs/simulation_experiments/run_YYYYMMDD_HHMMSS/
```

A run directory contains:

```text
README.md
run.log
simulation_summary.json
simple_parameter_comparison.tsv
general_parameter_comparison.tsv
step_length_results.tsv
simple_replicate_results.tsv
general_replicate_results.tsv
general_substitution_matrix_comparison.tsv
figures/
```

The output-specific `README.md` summarizes the configuration and explains the files in that run directory.

### Run log

```text
run.log
```

This file captures terminal output from the run, including miRBench messages and detailed optimizer progress when `--verbose` is used.

### JSON summary

```text
simulation_summary.json
```

This file contains the run configuration, dataset summary, per-experiment results, and parameter-comparison sections.

### Parameter recovery reports

```text
simple_parameter_comparison.tsv
general_parameter_comparison.tsv
```

Each table contains:

```text
parameter    true    estimated    absolute_error
```

These reports compare the simulated truth against the fitted EstimAlign parameters.

### Experiment result tables

```text
step_length_results.tsv
simple_replicate_results.tsv
general_replicate_results.tsv
general_substitution_matrix_comparison.tsv
```

These TSV files are designed to be easy to inspect in spreadsheet software or load into R, Python, or other analysis tools.

## How to interpret outputs

Start with the parameter recovery reports. The `true` column is the parameter value used to simulate labels. The `estimated` column is the value learned by EstimAlign. The `absolute_error` column is the recovery error. Smaller absolute error means the fitted model is closer to the simulated truth.

Use `step_length_results.tsv` to compare optimizer step sizes. Each row is one constant step length. Useful columns are `final_loglik`, `max_loglik`, and the estimated alignment parameters.

Use `simple_replicate_results.tsv` to inspect run-to-run variability for the simple match/mismatch scoring model.

Use `general_replicate_results.tsv` to inspect run-to-run variability for the general asymmetric substitution matrix model.

Use `general_substitution_matrix_comparison.tsv` for the general substitution matrix experiment. Each row is one matrix entry. The `char1` and `char2` columns identify the substitution, while `true`, `estimated`, and `absolute_error` compare the simulated and fitted substitution score.

Use `simulation_summary.json` when you need the complete machine-readable record of a run, including configuration, summary metrics, and all nested experiment outputs.

Use `run.log` when you need to inspect terminal messages or detailed optimizer progress.

### Figures

When plots are enabled, figures are written to:

```text
figures/
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
simple_replicate_loglikelihoods.png
general_replicate_loglikelihoods.png
general_scores_hist.png
general_logit_scores_hist.png
general_optimization_trajectory.png
general_substitution_comparison.png
```

The figures are exported at high resolution and formatted with consistent typography, axis labels, gridlines, and legends suitable for manuscript inspection.

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
src/estimalign/cli.py                          Command-line interface.
src/estimalign/simulation_experiments.py       Main orchestration for the simulation workflow.
src/estimalign/simulation_config.py            Dataclasses for run configuration and simulation truths.
src/estimalign/simulation_dataset.py           miRBench dataset loading.
src/estimalign/simulation_model_experiments.py Experiment-specific simulation steps.
src/estimalign/simulation_metrics.py           Scoring, likelihood, and parameter-comparison helpers.
src/estimalign/simulation_outputs.py           JSON, TSV, and output README writers.
src/estimalign/simulation_plots.py             Figure generation helpers.
```

The project is packaged through `pyproject.toml` and locked with `uv.lock`.

## Current scope

The current command-line workflow covers the non-protein simulated-interaction validation workflow. The real miRNA-target prediction case study is maintained separately and is not part of this repository workflow.

The protein experiments are not included in the main command-line workflow yet. They should be converted separately with explicit, configurable input paths before being added to the package.

## Suggested next improvements

- Convert protein experiments into separate configurable commands.
- Add small example output snippets for documentation.
