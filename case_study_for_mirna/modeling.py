from __future__ import annotations

import time

import numpy as np

from src.estimalign import estimalign
from src.optimization import create_constant_step, create_powerstep


def make_stepfunction(name: str, scale: float):
    if name == "power":
        return create_powerstep(scale=scale)
    if name == "constant":
        return create_constant_step(scale=scale)
    raise ValueError(f"Unknown stepfunction: {name}")


def fit_configuration(fit_inputs, config: dict):
    start_time = time.perf_counter()
    result = estimalign(
        seqlistA=fit_inputs[0],
        seqlistB=fit_inputs[1],
        labels=fit_inputs[2],
        aligner_mode=config["aligner_mode"],
        gap_mode=config["gap_mode"],
        substitution_mode=config["substitution_mode"],
        stepfunction=make_stepfunction(config["stepfunction"], float(config["step_scale"])),
        max_iter=int(config["max_iter"]),
        num_threads=int(config["num_threads"]),
        verbose=False,
    )
    return result, time.perf_counter() - start_time


def summarize_result(config: dict, result: dict, runtime_seconds: float) -> dict:
    row = {
        **config,
        "status": "ok",
        "error": "",
        "final_loglik": result.get("final_loglik", np.nan),
        "alpha": result.get("alpha", np.nan),
        "runtime_seconds": round(runtime_seconds, 3),
        "iterations_completed": len(result.get("subgradient_l2_trajectory", [])),
    }
    for key in ["open_gap_score", "extend_gap_score", "gap_score", "match_score", "mismatch_score"]:
        if key in result:
            row[key] = result[key]
    return row


def build_trajectory_rows(config: dict, result: dict) -> list[dict]:
    loglik = list(result.get("loglik_trajectory", []))
    subgrad = list(result.get("subgradient_l2_trajectory", []))
    rows = []
    for iteration in range(max(len(loglik), len(subgrad) + 1)):
        rows.append(
            {
                **config,
                "iteration": iteration,
                "loglik": loglik[iteration] if iteration < len(loglik) else np.nan,
                "subgradient_l2": subgrad[iteration - 1] if 0 < iteration <= len(subgrad) else np.nan,
            }
        )
    return rows


def rank_successful(summary_rows: list[dict], objective: str):
    successful = [row for row in summary_rows if row.get("status") == "ok"]
    if not successful:
        return []
    if objective == "final_loglik":
        key = lambda row: (row.get("final_loglik", -np.inf), row.get("ap_fit", -np.inf))
    else:
        key = lambda row: (row.get("ap_validation", -np.inf), row.get("roc_auc_validation", -np.inf), row.get("ap_fit", -np.inf))
    return sorted(successful, key=key, reverse=True)
