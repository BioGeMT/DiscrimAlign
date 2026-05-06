from __future__ import annotations

from typing import Any

import numpy as np
from Bio.Align import PairwiseAligner


def score_sequence_pairs(
    aligner: PairwiseAligner,
    first_sequences: list[Any],
    second_sequences: list[Any],
) -> np.ndarray:
    return np.array(
        [
            aligner.score(first_sequence, second_sequence)
            for first_sequence, second_sequence in zip(
                first_sequences,
                second_sequences,
            )
        ]
    )


def sample_labels(logit_scores: np.ndarray) -> np.ndarray:
    from numpy import random as rd

    return rd.rand(len(logit_scores)) <= logit_scores


def compute_loglik(logit_scores: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.sum(np.log(logit_scores[labels]))
        + np.sum(np.log(1 - logit_scores[~labels]))
    )


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def compare_named_parameters(
    *,
    truth: dict[str, float],
    estimates: dict[str, Any],
) -> list[dict[str, float | str | None]]:
    rows = []
    for parameter, true_value in truth.items():
        estimated_value = as_float(estimates.get(parameter))
        rows.append(
            {
                "parameter": parameter,
                "true": float(true_value),
                "estimated": estimated_value,
                "absolute_error": abs(float(true_value) - estimated_value)
                if estimated_value is not None
                else None,
            }
        )
    return rows


def compare_substitution_matrices(
    true_matrix: Any,
    estimated_matrix: Any,
) -> dict[str, Any]:
    rows = []
    true_values = []
    estimated_values = []

    for char1 in true_matrix.alphabet:
        for char2 in true_matrix.alphabet:
            true_value = float(true_matrix[char1, char2])
            estimated_value = float(estimated_matrix[char1, char2])
            rows.append(
                {
                    "char1": char1,
                    "char2": char2,
                    "true": true_value,
                    "estimated": estimated_value,
                    "absolute_error": abs(true_value - estimated_value),
                }
            )
            true_values.append(true_value)
            estimated_values.append(estimated_value)

    correlation = float(np.corrcoef(true_values, estimated_values)[0, 1])
    mean_absolute_error = float(
        np.mean(np.abs(np.array(true_values) - np.array(estimated_values)))
    )

    return {
        "rows": rows,
        "correlation": correlation,
        "mean_absolute_error": mean_absolute_error,
    }


def summarize_params(params: dict[str, Any]) -> dict[str, Any]:
    loglik_trajectory = params.get("loglik_trajectory", [])

    summary = {
        "final_loglik": as_float(params.get("final_loglik")),
        "max_loglik": as_float(max(loglik_trajectory))
        if len(loglik_trajectory) > 0
        else None,
        "alpha": as_float(params.get("alpha")),
        "match_score": as_float(params.get("match_score")),
        "mismatch_score": as_float(params.get("mismatch_score")),
        "open_gap_score": as_float(params.get("open_gap_score")),
        "extend_gap_score": as_float(params.get("extend_gap_score")),
    }

    return {
        key: value
        for key, value in summary.items()
        if value is not None
    }
