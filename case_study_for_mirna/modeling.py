from __future__ import annotations

import pickle
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.discrimalign import discrimalign
from src.optimization import create_constant_step, create_powerstep

SKLEARN_PENALTY_WARNING = "'penalty' was deprecated in version 1.8"


def make_stepfunction(name: str, scale: float):
    if name == "power":
        return create_powerstep(scale=scale)
    if name == "constant":
        return create_constant_step(scale=scale)
    raise ValueError(f"Unknown stepfunction: {name}")


def load_warm_start_model(path: str | Path, config: dict):
    """Load a saved case-study model artifact for warm-start continuation."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if "aligner" not in payload:
        raise ValueError(f"Warm-start model {path} does not contain an aligner.")

    aligner = deepcopy(payload["aligner"])
    summary = payload.get("summary", {})
    saved_config = payload.get("config", {})

    for key in ["aligner_mode", "gap_mode", "substitution_mode"]:
        saved_value = saved_config.get(key) or summary.get(key)
        requested_value = config.get(key)
        if saved_value is not None and requested_value is not None and str(saved_value) != str(requested_value):
            raise ValueError(
                f"Warm-start model {path} has {key}={saved_value!r}, "
                f"but this run requested {requested_value!r}."
            )

    parameters = {"alpha": summary.get("alpha", payload.get("alpha"))}
    if parameters["alpha"] is None:
        raise ValueError(f"Warm-start model {path} does not contain alpha in its summary.")

    if config["gap_mode"] == "affine":
        parameters["open_gap_score"] = getattr(aligner, "open_gap_score")
        parameters["extend_gap_score"] = getattr(aligner, "extend_gap_score")
    else:
        parameters["gap_score"] = getattr(aligner, "gap_score")

    if config["substitution_mode"] == "simple":
        parameters["match_score"] = getattr(aligner, "match_score")
        parameters["mismatch_score"] = getattr(aligner, "mismatch_score")
    else:
        parameters["substitution_matrix"] = deepcopy(getattr(aligner, "substitution_matrix"))

    return aligner, parameters


def fit_configuration(fit_inputs, config: dict):
    start_time = time.perf_counter()
    warm_start_model = config.get("warm_start_model")
    baseline_aligner = None
    initial_parameters = None
    if warm_start_model:
        baseline_aligner, initial_parameters = load_warm_start_model(warm_start_model, config)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=SKLEARN_PENALTY_WARNING,
            category=FutureWarning,
            module="sklearn.linear_model._logistic",
        )
        result = discrimalign(
            seqlistA=fit_inputs[0],
            seqlistB=fit_inputs[1],
            labels=fit_inputs[2],
            baseline_aligner=baseline_aligner,
            aligner_mode=config["aligner_mode"],
            gap_mode=config["gap_mode"],
            substitution_mode=config["substitution_mode"],
            stepfunction=make_stepfunction(config["stepfunction"], float(config["step_scale"])),
            max_iter=int(config["max_iter"]),
            num_threads=int(config["num_threads"]),
            subgradient_scale=1.0 / len(fit_inputs[2]),
            initial_parameters=initial_parameters,
            return_alignments=False,
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
