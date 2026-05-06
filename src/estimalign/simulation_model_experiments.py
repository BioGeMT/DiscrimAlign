from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from estimalign.estimalign import estimalign
from estimalign.logit_link import logit_partial_scores
from estimalign.optimization import create_constant_step
from estimalign.simulation_config import AlphaMode, GeneralMatrixTruth, SimpleModelTruth
from estimalign.simulation_metrics import (
    compare_named_parameters,
    compare_substitution_matrices,
    compute_loglik,
    sample_labels,
    score_sequence_pairs,
    summarize_params,
)
from estimalign.simulation_plots import (
    plot_histogram,
    plot_label_scatter,
    plot_optimization_trajectories,
    plot_replicate_loglikelihoods,
    plot_step_length_loglikelihoods,
    plot_substitution_comparison,
)


def select_alpha(scores: np.ndarray, *, fixed_alpha: float, alpha_mode: AlphaMode) -> float:
    if alpha_mode == "fixed":
        return fixed_alpha
    if alpha_mode == "negative-median":
        return float(-np.median(scores))
    raise ValueError(f"Unsupported alpha mode: {alpha_mode}")


def build_simple_aligner(truth: SimpleModelTruth | None = None) -> PairwiseAligner:
    truth = truth or SimpleModelTruth()
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = truth.gap_open
    aligner.extend_gap_score = truth.gap_extend
    aligner.match = truth.match
    aligner.mismatch = truth.mismatch
    return aligner


def build_general_aligner(truth: GeneralMatrixTruth | None = None) -> tuple[PairwiseAligner, Any]:
    truth = truth or GeneralMatrixTruth()
    substitution = substitution_matrices.Array(
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
    aligner.substitution_matrix = substitution
    return aligner, substitution


def run_simple_mirna_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    max_iter: int,
    step_length: float,
    alpha_mode: AlphaMode,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
    verbose: bool,
) -> dict[str, Any]:
    truth = SimpleModelTruth()
    aligner = build_simple_aligner(truth)
    scores = score_sequence_pairs(aligner, mirna_sequences, gene_sequences)
    alpha = select_alpha(scores, fixed_alpha=truth.alpha, alpha_mode=alpha_mode)
    logit_scores = logit_partial_scores(scores, alpha)
    labels = sample_labels(logit_scores)
    true_loglik = compute_loglik(logit_scores, labels)

    params = estimalign(
        mirna_sequences,
        gene_sequences,
        labels,
        stepfunction=create_constant_step(step_length),
        aligner_mode="local",
        substitution_mode="simple",
        gap_mode="affine",
        verbose=verbose,
        max_iter=max_iter,
        stochastic_factor=stochastic_factor,
        num_threads=num_threads,
    )

    if make_plots:
        plot_histogram(scores, figures_dir / "simple_scores_hist.png", title="Simple model alignment scores")
        plot_histogram(logit_scores, figures_dir / "simple_logit_scores_hist.png", title="Simple model logit scores")
        plot_label_scatter(scores, labels, figures_dir / "simple_scores_vs_labels.png", title="Simple model scores vs labels", xlabel="Alignment score")
        plot_label_scatter(logit_scores, labels, figures_dir / "simple_logit_scores_vs_labels.png", title="Simple model logit scores vs labels", xlabel="Logit score")
        plot_optimization_trajectories(params, max_iter=max_iter, true_loglik=true_loglik, output_path=figures_dir / "simple_optimization_trajectory.png")

    return {
        "truth": truth,
        "alpha_mode": alpha_mode,
        "simulation_alpha": alpha,
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
    verbose: bool,
) -> dict[str, Any]:
    labels = sample_labels(logit_scores)
    step_lengths = np.linspace(0.000005, 0.00005, num=10)
    runs = []
    raw_results = []

    for step_length in step_lengths:
        params = estimalign(
            mirna_sequences,
            gene_sequences,
            labels,
            stepfunction=create_constant_step(float(step_length)),
            aligner_mode="local",
            substitution_mode="simple",
            gap_mode="affine",
            verbose=verbose,
            max_iter=max_iter,
            stochastic_factor=stochastic_factor,
            num_threads=num_threads,
        )
        raw_results.append(params)
        runs.append({"step_length": float(step_length), **summarize_params(params)})

    if make_plots:
        plot_step_length_loglikelihoods(raw_results, step_lengths, max_iter=max_iter, true_loglik=true_loglik, output_path=figures_dir / "step_length_loglikelihoods.png")
        plot_step_length_loglikelihoods(raw_results, step_lengths, max_iter=max_iter, true_loglik=true_loglik, output_path=figures_dir / "step_length_loglikelihoods_zoom.png", xlim=(0, min(5, max_iter)))

    return {"true_loglik": true_loglik, "runs": runs}


def run_simple_replicate_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    logit_scores: np.ndarray,
    replicate_count: int,
    max_iter: int,
    step_length: float,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
    verbose: bool,
) -> dict[str, Any]:
    return run_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=logit_scores,
        replicate_count=replicate_count,
        max_iter=max_iter,
        step_length=step_length,
        substitution_mode="simple",
        figure_name="simple_replicate_loglikelihoods.png",
        stochastic_factor=stochastic_factor,
        num_threads=num_threads,
        figures_dir=figures_dir,
        make_plots=make_plots,
        verbose=verbose,
    )


def run_general_replicate_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    logit_scores: np.ndarray,
    replicate_count: int,
    max_iter: int,
    step_length: float,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
    verbose: bool,
) -> dict[str, Any]:
    return run_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=logit_scores,
        replicate_count=replicate_count,
        max_iter=max_iter,
        step_length=step_length,
        substitution_mode="general",
        figure_name="general_replicate_loglikelihoods.png",
        stochastic_factor=stochastic_factor,
        num_threads=num_threads,
        figures_dir=figures_dir,
        make_plots=make_plots,
        verbose=verbose,
    )


def run_replicate_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    logit_scores: np.ndarray,
    replicate_count: int,
    max_iter: int,
    step_length: float,
    substitution_mode: str,
    figure_name: str,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
    verbose: bool,
) -> dict[str, Any]:
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
            stepfunction=create_constant_step(step_length),
            aligner_mode="local",
            substitution_mode=substitution_mode,
            gap_mode="affine",
            verbose=verbose,
            max_iter=max_iter,
            stochastic_factor=stochastic_factor,
            num_threads=num_threads,
        )
        run_summary = {"replicate_index": replicate_index, "true_loglik": true_loglik, **summarize_params(params)}
        raw_results.append(params)
        runs.append(run_summary)
        true_logliks.append(true_loglik)
        if run_summary.get("final_loglik") is not None:
            final_logliks.append(run_summary["final_loglik"])

    if make_plots:
        plot_replicate_loglikelihoods(raw_results, true_logliks, max_iter=max_iter, output_path=figures_dir / figure_name)

    return {
        "replicate_count": replicate_count,
        "max_iter": max_iter,
        "mean_true_loglik": float(np.mean(true_logliks)) if true_logliks else None,
        "mean_final_loglik": float(np.mean(final_logliks)) if final_logliks else None,
        "true_logliks": true_logliks,
        "final_logliks": final_logliks,
        "runs": runs,
    }


def run_general_matrix_experiment(
    mirna_sequences: list[Seq],
    gene_sequences: list[Seq],
    *,
    max_iter: int,
    step_length: float,
    alpha_mode: AlphaMode,
    stochastic_factor: float,
    num_threads: int,
    figures_dir: Path,
    make_plots: bool,
    verbose: bool,
) -> dict[str, Any]:
    truth = GeneralMatrixTruth()
    aligner, true_substitution = build_general_aligner(truth)
    scores = score_sequence_pairs(aligner, mirna_sequences, gene_sequences)
    alpha = select_alpha(scores, fixed_alpha=truth.alpha, alpha_mode=alpha_mode)
    logit_scores = logit_partial_scores(scores, alpha)
    labels = sample_labels(logit_scores)
    true_loglik = compute_loglik(logit_scores, labels)

    params = estimalign(
        mirna_sequences,
        gene_sequences,
        labels,
        stepfunction=create_constant_step(step_length),
        aligner_mode="local",
        substitution_mode="general",
        gap_mode="affine",
        stochastic_factor=stochastic_factor,
        verbose=verbose,
        max_iter=max_iter,
        num_threads=num_threads,
    )

    substitution_comparison = compare_substitution_matrices(true_substitution, params["substitution_matrix"])
    parameter_comparison = compare_named_parameters(
        truth={"alpha": alpha, "open_gap_score": truth.gap_open, "extend_gap_score": truth.gap_extend},
        estimates=params,
    )

    if make_plots:
        plot_histogram(scores, figures_dir / "general_scores_hist.png", title="General matrix alignment scores")
        plot_histogram(logit_scores, figures_dir / "general_logit_scores_hist.png", title="General matrix logit scores")
        plot_optimization_trajectories(params, max_iter=max_iter, true_loglik=true_loglik, output_path=figures_dir / "general_optimization_trajectory.png")
        plot_substitution_comparison(substitution_comparison, figures_dir / "general_substitution_comparison.png")

    return {
        "truth": {"gap_open": truth.gap_open, "gap_extend": truth.gap_extend, "fixed_alpha": truth.alpha},
        "alpha_mode": alpha_mode,
        "simulation_alpha": alpha,
        "logit_scores": logit_scores,
        "true_loglik": true_loglik,
        **summarize_params(params),
        "parameter_comparison": parameter_comparison,
        "substitution_correlation": substitution_comparison["correlation"],
        "substitution_mean_absolute_error": substitution_comparison["mean_absolute_error"],
        "substitution_comparison": substitution_comparison["rows"],
    }


def summarize_simple_model(result: dict[str, Any]) -> dict[str, Any]:
    truth = result["truth"]
    params = result["params"]
    parameter_comparison = compare_named_parameters(
        truth={
            "alpha": result["simulation_alpha"],
            "match_score": truth.match,
            "mismatch_score": truth.mismatch,
            "open_gap_score": truth.gap_open,
            "extend_gap_score": truth.gap_extend,
        },
        estimates=params,
    )

    return {
        "true_loglik": result["true_loglik"],
        "alpha_mode": result["alpha_mode"],
        "simulation_alpha": result["simulation_alpha"],
        **summarize_params(params),
        "parameter_comparison": parameter_comparison,
    }
