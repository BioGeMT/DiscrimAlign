from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_simulation_outputs(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    write_json(output_dir / "simulation_summary.json", summary)
    write_simulation_tsv_outputs(output_dir, summary)
    write_output_readme(output_dir, summary)


def write_simulation_tsv_outputs(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    write_tsv(
        output_dir / "simple_parameter_comparison.tsv",
        summary["simple_model"]["parameter_comparison"],
        fieldnames=["parameter", "true", "estimated", "absolute_error"],
    )
    write_tsv(
        output_dir / "step_length_results.tsv",
        summary["step_length_experiment"]["runs"],
        fieldnames=[
            "step_length",
            "final_loglik",
            "max_loglik",
            "alpha",
            "match_score",
            "mismatch_score",
            "open_gap_score",
            "extend_gap_score",
        ],
    )
    write_replicate_tsv(
        output_dir / "simple_replicate_results.tsv",
        summary["simple_replicate_experiment"]["runs"],
    )
    write_replicate_tsv(
        output_dir / "general_replicate_results.tsv",
        summary["general_replicate_experiment"]["runs"],
    )
    write_tsv(
        output_dir / "general_parameter_comparison.tsv",
        summary["general_matrix_experiment"]["parameter_comparison"],
        fieldnames=["parameter", "true", "estimated", "absolute_error"],
    )
    write_tsv(
        output_dir / "general_substitution_matrix_comparison.tsv",
        summary["general_matrix_experiment"]["substitution_comparison"],
        fieldnames=["char1", "char2", "true", "estimated", "absolute_error"],
    )


def write_output_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    config = summary["config"]
    dataset = summary["dataset"]
    content = f"""# EstimAlign simulation run

This directory contains one complete EstimAlign simulation run.

## Run metadata

- Run directory: `{config.get('run_dir', output_dir)}`
- Output root: `{config.get('output_root', output_dir.parent)}`
- Dataset index: `{dataset['dataset_index']}`
- Dataset split: `{dataset['split']}`
- Sequence pairs: `{dataset['sequence_count']}`
- Alpha mode: `{config['alpha_mode']}`
- Simple iterations: `{config['simple_max_iter']}`
- General matrix iterations: `{config['general_max_iter']}`
- Step-length iterations: `{config['step_max_iter']}`
- Replicate iterations: `{config['replicate_max_iter']}`
- Replicate count: `{config['replicate_count']}`

## Files

| File | Purpose |
| --- | --- |
| `simulation_summary.json` | Complete machine-readable record of the run. |
| `run.log` | Terminal output captured during the run. Use `--verbose` to include detailed optimizer progress. |
| `simple_parameter_comparison.tsv` | True vs estimated parameters for the simple scoring model. |
| `general_parameter_comparison.tsv` | True vs estimated parameters for the general asymmetric matrix model. |
| `step_length_results.tsv` | Optimizer behavior across constant step lengths. |
| `simple_replicate_results.tsv` | Replicate runs for the simple scoring model. |
| `general_replicate_results.tsv` | Replicate runs for the general asymmetric matrix model. |
| `general_substitution_matrix_comparison.tsv` | Entrywise true vs estimated substitution scores. |
| `figures/` | PNG figures, when plot generation is enabled. |

## Reading the tables

For parameter-recovery tables, the `true` column is the simulated value, `estimated` is the fitted EstimAlign value, and `absolute_error` is the recovery error.

For replicate tables, each row is one independently sampled label vector fitted with the same model class.

For the substitution matrix comparison, each row corresponds to one ordered character pair `(char1, char2)`.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_replicate_tsv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    write_tsv(
        output_path,
        rows,
        fieldnames=[
            "replicate_index",
            "true_loglik",
            "final_loglik",
            "max_loglik",
            "alpha",
            "match_score",
            "mismatch_score",
            "open_gap_score",
            "extend_gap_score",
        ],
    )


def write_tsv(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
