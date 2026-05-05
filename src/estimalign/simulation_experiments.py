from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from miRBench.dataset import get_dataset_df, list_datasets
from numpy import random as rd

from estimalign.estimalign import estimalign
from estimalign.logit_link import logit_partial_scores
from estimalign.optimization import create_constant_step


@dataclass(frozen=True)
class SimulationConfig:
    output_dir: Path
    dataset_index: int = 0
    split: str = "train"
    random_seed: int = 7
    max_records: int | None = None
    num_threads: int = 16
    stochastic_factor: float = 0.001
    simple_max_iter: int = 50
    step_max_iter: int = 10
    replicate_max_iter: int = 5
    replicate_count: int = 20
    make_plots: bool = True


@dataclass(frozen=True)
class SimpleModelTruth:
    match: float = 1.0
    mismatch: float = -1.0
    gap_open: float = -1.2
    gap_extend: float = -0.1
    alpha: float = -9.0


def run_simulation_experiments(
    output_dir: Path,
    *,
    dataset_index: int = 0,
    split: str = "train",
    random_seed: int = 7,
    max_records: int | None = None,
    num_threads: int = 16,
    stochastic_factor: float = 0.001,
    simple_max_iter: int = 50,
    step_max_iter: int = 10,
    replicate_max_iter: int = 5,
    replicate_count: int = 20,
    make_plots: bool = True,
) -> dict[str, Any]:
    config = SimulationConfig(
        output_dir=output_dir,
        dataset_index=dataset_index,
        split=split,
        random_seed=random_seed,
        max_records=max_records,
        num_threads=num_threads,
        stochastic_factor=stochastic_factor,
        simple_max_iter=simple_max_iter,
        step_max_iter=step_max_iter,
        replicate_max_iter=replicate_max_iter,
        replicate_count=replicate_count,
        make_plots=make_plots,
    )
    return run_mirna_simulation_suite(config)


def run_mirna_simulation_suite(config: SimulationConfig) -> dict[str, Any]:
    rd.seed(config.random_seed)
    np.random.seed(config.random_seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    mirna_sequences, gene_sequences = load_mirbench_sequences(
        dataset_index=config.dataset_index,
        split=config.split,
        max_records=config.max_records,
    )

    simple_result = run_simple_mirna_experiment(
        mirna_sequences,
        gene_sequences,
        max_iter=config.simple_max_iter,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
    )

    step_result = run_step_length_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=simple_result["logit_scores"],
        true_loglik=simple_result["true_loglik"],
        max_iter=config.step_max_iter,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
    )

    replicate_result = run_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=simple_result["logit_scores"],
        replicate_count=config.replicate_count,
        max_iter=config.replicate_max_iter,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
    )

    summary = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "dataset": {
            "sequence_count": len(mirna_sequences),
            "dataset_index": config.dataset_index,
            "split": config.split,
        },
        "simple_model": summarize_simple_model(simple_result),
        "step_length_experiment": step_result,
        "replicate_experiment": replicate_result,
    }

    write_json(config.output_dir / "simulation_summary.json", summary)
    return summary


def load_mirbench_sequences(
    *,
    dataset_index: int,
    split: str,
    max_records: int | None,
) -> tuple[list[Seq], list[Seq]]:
    datasets = list_datasets()
    dataset_name = datasets[dataset_index]
    dataset = get_dataset_df(dataset_name, split=split)

    if max_records is not None:
        dataset = dataset.head(max_records)

    mirna_sequences = [Seq(seq) for seq in dataset["noncodingRNA"]]
    gene_sequences = [
        Seq(seq).reverse_complement()
        for seq in dataset["gene"]
    ]
    return mirna_sequences, gene_sequences


def run_simple_mirna_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    max_iter: int,
    stochastic_factor: float,
    num_threads: int,
) -> dict[str, Any]:
    truth = SimpleModelTruth()

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = truth.gap_open
    aligner.extend_gap_score = truth.gap_extend
    aligner.match = truth.match
    aligner.mismatch = truth.mismatch

    scores = score_sequence_pairs(
        aligner,
        mirna_sequences,
        gene_sequences,
    )

    logit_scores = logit_partial_scores(scores, truth.alpha)
    labels = sample_labels(logit_scores)
    true_loglik = compute_loglik(logit_scores, labels)

    const_step = create_constant_step(0.00001)
    params = estimalign(
        mirna_sequences,
        gene_sequences,
        labels,
        stepfunction=const_step,
        aligner_mode="local",
        substitution_mode="simple",
        gap_mode="affine",
        verbose=True,
        max_iter=max_iter,
        stochastic_factor=stochastic_factor,
        num_threads=num_threads,
    )

    return {
        "truth": truth,
        "scores": scores,
        "logit_scores": logit_scores,
        "labels": labels,
        "true_loglik": true_loglik,
        "params": params,
    }


def run_step_length_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    logit_scores: np.ndarray,
    true_loglik: float,
    max_iter: int,
    stochastic_factor: float,
    num_threads: int,
) -> dict[str, Any]:
    labels = sample_labels(logit_scores)
    step_lengths = np.linspace(0.000005, 0.00005, num=10)

    runs = []
    for step_length in step_lengths:
        const_step = create_constant_step(float(step_length))
        params = estimalign(
            mirna_sequences,
            gene_sequences,
            labels,
            stepfunction=const_step,
            aligner_mode="local",
            substitution_mode="simple",
            gap_mode="affine",
            verbose=False,
            max_iter=max_iter,
            stochastic_factor=stochastic_factor,
            num_threads=num_threads,
        )

        runs.append(
            {
                "step_length": float(step_length),
                **summarize_params(params),
            }
        )

    return {
        "true_loglik": true_loglik,
        "runs": runs,
    }


def run_replicate_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    logit_scores: np.ndarray,
    replicate_count: int,
    max_iter: int,
    stochastic_factor: float,
    num_threads: int,
) -> dict[str, Any]:
    const_step = create_constant_step(0.00001)

    runs = []
    true_logliks = []
    final_logliks = []

    for replicate_index in range(replicate_count):
        labels = sample_labels(logit_scores)
        true_loglik = compute_loglik(logit_scores, labels)

        params = estimalign(
            mirna_sequences,
            gene_sequences,
            labels,
            stepfunction=const_step,
            aligner_mode="local",
            substitution_mode="simple",
            gap_mode="affine",
            verbose=False,
            max_iter=max_iter,
            stochastic_factor=stochastic_factor,
            num_threads=num_threads,
        )

        run_summary = {
            "replicate_index": replicate_index,
            "true_loglik": true_loglik,
            **summarize_params(params),
        }

        runs.append(run_summary)
        true_logliks.append(true_loglik)

        final_loglik = run_summary.get("final_loglik")
        if final_loglik is not None:
            final_logliks.append(final_loglik)

    return {
        "replicate_count": replicate_count,
        "max_iter": max_iter,
        "mean_true_loglik": float(np.mean(true_logliks))
        if true_logliks
        else None,
        "mean_final_loglik": float(np.mean(final_logliks))
        if final_logliks
        else None,
        "true_logliks": true_logliks,
        "final_logliks": final_logliks,
        "runs": runs,
    }


def score_sequence_pairs(
    aligner: PairwiseAligner,
    first_sequences: list[Seq],
    second_sequences: list[Seq],
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
    return rd.rand(len(logit_scores)) <= logit_scores


def compute_loglik(logit_scores: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.sum(np.log(logit_scores[labels]))
        + np.sum(np.log(1 - logit_scores[~labels]))
    )


def summarize_simple_model(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "true_loglik": result["true_loglik"],
        **summarize_params(result["params"]),
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


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