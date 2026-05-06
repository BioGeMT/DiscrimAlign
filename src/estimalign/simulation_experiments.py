from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from matplotlib import pyplot as plt
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


@dataclass(frozen=True)
class GeneralMatrixTruth:
    gap_open: float = -1.2
    gap_extend: float = -0.1
    alpha: float = -12.0


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
    figures_dir = config.output_dir / "figures"
    if config.make_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)

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
        figures_dir=figures_dir,
        make_plots=config.make_plots,
    )

    step_result = run_step_length_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=simple_result["logit_scores"],
        true_loglik=simple_result["true_loglik"],
        max_iter=config.step_max_iter,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
    )

    replicate_result = run_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=simple_result["logit_scores"],
        replicate_count=config.replicate_count,
        max_iter=config.replicate_max_iter,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
    )

    general_matrix_result = run_general_matrix_experiment(
        mirna_sequences,
        gene_sequences,
        max_iter=config.simple_max_iter,
        stochastic_factor=0.01,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
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
        "general_matrix_experiment": general_matrix_result,
    }

    write_json(config.output_dir / "simulation_summary.json", summary)
    write_simulation_tsv_outputs(config.output_dir, summary)
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
    figures_dir: Path,
    make_plots: bool,
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

    if make_plots:
        plot_histogram(
            scores,
            figures_dir / "simple_scores_hist.png",
            title="Simple model alignment scores",
        )
        plot_histogram(
            logit_scores,
            figures_dir / "simple_logit_scores_hist.png",
            title="Simple model logit scores",
        )
        plot_label_scatter(
            scores,
            labels,
            figures_dir / "simple_scores_vs_labels.png",
            title="Simple model scores vs labels",
            xlabel="Alignment score",
        )
        plot_label_scatter(
            logit_scores,
            labels,
            figures_dir / "simple_logit_scores_vs_labels.png",
            title="Simple model logit scores vs labels",
            xlabel="Logit score",
        )
        plot_optimization_trajectories(
            params,
            max_iter=max_iter,
            true_loglik=true_loglik,
            output_path=figures_dir / "simple_optimization_trajectory.png",
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
    figures_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    labels = sample_labels(logit_scores)
    step_lengths = np.linspace(0.000005, 0.00005, num=10)

    runs = []
    raw_results = []
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

        raw_results.append(params)
        runs.append(
            {
                "step_length": float(step_length),
                **summarize_params(params),
            }
        )

    if make_plots:
        plot_step_length_loglikelihoods(
            raw_results,
            step_lengths,
            max_iter=max_iter,
            true_loglik=true_loglik,
            output_path=figures_dir / "step_length_loglikelihoods.png",
        )
        plot_step_length_loglikelihoods(
            raw_results,
            step_lengths,
            max_iter=max_iter,
            true_loglik=true_loglik,
            output_path=figures_dir / "step_length_loglikelihoods_zoom.png",
            xlim=(0, min(5, max_iter)),
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
    figures_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    const_step = create_constant_step(0.00001)

    runs = []
    raw_results = []
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

        raw_results.append(params)
        runs.append(run_summary)
        true_logliks.append(true_loglik)

        final_loglik = run_summary.get("final_loglik")
        if final_loglik is not None:
            final_logliks.append(final_loglik)

    if make_plots:
        plot_replicate_loglikelihoods(
            raw_results,
            true_logliks,
            max_iter=max_iter,
            output_path=figures_dir / "replicate_loglikelihoods.png",
        )

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


def run_general_matrix_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    max_iter: int,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    truth = GeneralMatrixTruth()

    true_substitution = substitution_matrices.Array(
        alphabet="ACTG",
        data=np.array(
            [
                [1.0, -0.3, -1.0, -0.8],
                [-0.6, 1.2, -0.3, -1.0],
                [-1.2, -0.4, 1.0, -0.8],
                [-0.4, -1.4, -0.9, 1.3],
            ]
        ),
    )

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = truth.gap_open
    aligner.extend_gap_score = truth.gap_extend
    aligner.substitution_matrix = true_substitution

    scores = score_sequence_pairs(
        aligner,
        mirna_sequences,
        gene_sequences,
    )

    logit_scores = logit_partial_scores(scores, truth.alpha)
    labels = sample_labels(logit_scores)
    true_loglik = compute_loglik(logit_scores, labels)

    const_step = create_constant_step(0.00005)
    params = estimalign(
        mirna_sequences,
        gene_sequences,
        labels,
        stepfunction=const_step,
        aligner_mode="local",
        substitution_mode="general",
        gap_mode="affine",
        stochastic_factor=stochastic_factor,
        verbose=True,
        max_iter=max_iter,
        num_threads=num_threads,
    )

    substitution_comparison = compare_substitution_matrices(
        true_substitution,
        params["substitution_matrix"],
    )

    parameter_comparison = compare_named_parameters(
        truth={
            "alpha": truth.alpha,
            "open_gap_score": truth.gap_open,
            "extend_gap_score": truth.gap_extend,
        },
        estimates=params,
    )

    if make_plots:
        plot_histogram(
            scores,
            figures_dir / "general_scores_hist.png",
            title="General matrix alignment scores",
        )
        plot_histogram(
            logit_scores,
            figures_dir / "general_logit_scores_hist.png",
            title="General matrix logit scores",
        )
        plot_optimization_trajectories(
            params,
            max_iter=max_iter,
            true_loglik=true_loglik,
            output_path=figures_dir / "general_optimization_trajectory.png",
        )
        plot_substitution_comparison(
            substitution_comparison,
            figures_dir / "general_substitution_comparison.png",
        )

    return {
        "truth": {
            "gap_open": truth.gap_open,
            "gap_extend": truth.gap_extend,
            "alpha": truth.alpha,
        },
        "true_loglik": true_loglik,
        **summarize_params(params),
        "parameter_comparison": parameter_comparison,
        "substitution_correlation": substitution_comparison["correlation"],
        "substitution_mean_absolute_error": substitution_comparison[
            "mean_absolute_error"
        ],
        "substitution_comparison": substitution_comparison["rows"],
    }


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
    truth = result["truth"]
    params = result["params"]
    parameter_comparison = compare_named_parameters(
        truth={
            "alpha": truth.alpha,
            "match_score": truth.match,
            "mismatch_score": truth.mismatch,
            "open_gap_score": truth.gap_open,
            "extend_gap_score": truth.gap_extend,
        },
        estimates=params,
    )

    return {
        "true_loglik": result["true_loglik"],
        **summarize_params(params),
        "parameter_comparison": parameter_comparison,
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


def write_simulation_tsv_outputs(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    write_tsv(
        output_dir / "simple_parameter_comparison.tsv",
        summary["simple_model"]["parameter_comparison"],
        fieldnames=[
            "parameter",
            "true",
            "estimated",
            "absolute_error",
        ],
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

    write_tsv(
        output_dir / "replicate_results.tsv",
        summary["replicate_experiment"]["runs"],
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

    write_tsv(
        output_dir / "general_parameter_comparison.tsv",
        summary["general_matrix_experiment"]["parameter_comparison"],
        fieldnames=[
            "parameter",
            "true",
            "estimated",
            "absolute_error",
        ],
    )

    write_tsv(
        output_dir / "general_matrix_comparison.tsv",
        summary["general_matrix_experiment"]["substitution_comparison"],
        fieldnames=[
            "char1",
            "char2",
            "true",
            "estimated",
            "absolute_error",
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


def plot_histogram(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(values, bins=100)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_label_scatter(
    x_values: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x_values, labels, ".", alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel("Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_optimization_trajectories(
    params: dict[str, Any],
    *,
    max_iter: int,
    true_loglik: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.subplot(221)
    plt.plot(np.arange(max_iter), params["subgradient_l2_trajectory"])
    plt.plot([0, max_iter], [0, 0], "--")
    plt.title("Subgradient L2 norm trajectory")

    plt.subplot(222)
    plt.plot(np.arange(max_iter + 1), params["loglik_trajectory"])
    plt.plot([0, max_iter + 1], [true_loglik, true_loglik], "--")
    plt.title("LogLikelihood trajectory")

    plt.subplot(223)
    plt.plot(
        np.arange(max_iter // 2, max_iter),
        params["subgradient_l2_trajectory"][max_iter // 2 :],
    )
    plt.plot([max_iter // 2, max_iter], [0, 0], "--")
    plt.title("Subgradient L2 norm trajectory")

    plt.subplot(224)
    plt.plot(
        np.arange(max_iter // 2, max_iter + 1),
        params["loglik_trajectory"][max_iter // 2 :],
    )
    plt.plot([max_iter // 2, max_iter + 1], [true_loglik, true_loglik], "--")
    plt.title("LogLikelihood trajectory")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_step_length_loglikelihoods(
    results: list[dict[str, Any]],
    step_lengths: np.ndarray,
    *,
    max_iter: int,
    true_loglik: float,
    output_path: Path,
    xlim: tuple[int, int] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for params in results:
        plt.plot(
            np.arange(max_iter + 1),
            params["loglik_trajectory"],
            alpha=0.5,
        )

    plt.plot([0, max_iter], [true_loglik, true_loglik], "--")
    plt.legend([str(value) for value in step_lengths])

    if xlim is not None:
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_replicate_loglikelihoods(
    results: list[dict[str, Any]],
    true_logliks: list[float],
    *,
    max_iter: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 2.1))

    plt.subplot(121)
    for params in results:
        plt.plot(
            np.arange(max_iter + 1),
            params["loglik_trajectory"],
            alpha=0.2,
        )
    plt.title("Replicate loglikelihoods")

    plt.subplot(122)
    plt.plot([0, max_iter], [0, 0], "--")
    for params, true_loglik in zip(results, true_logliks):
        plt.plot(
            np.arange(max_iter + 1),
            true_loglik - params["loglik_trajectory"],
            alpha=0.2,
        )
    plt.title("True minus fitted loglikelihood")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_substitution_comparison(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = comparison["rows"]

    plt.figure()
    plt.plot(
        [row["true"] for row in rows],
        [row["estimated"] for row in rows],
        ".",
    )
    plt.xlabel("True substitution score")
    plt.ylabel("Estimated substitution score")
    plt.title("General matrix substitution comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


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
