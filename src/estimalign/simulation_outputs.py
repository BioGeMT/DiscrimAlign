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
