from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy import random as rd

from estimalign.simulation_config import AlphaMode, SimulationConfig
from estimalign.simulation_dataset import load_mirbench_sequences
from estimalign.simulation_model_experiments import (
    run_general_matrix_experiment,
    run_general_replicate_experiment,
    run_simple_mirna_experiment,
    run_simple_replicate_experiment,
    run_step_length_experiment,
    summarize_simple_model,
)
from estimalign.simulation_outputs import write_simulation_outputs


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
    general_max_iter: int = 200,
    step_max_iter: int = 10,
    replicate_max_iter: int = 5,
    replicate_count: int = 20,
    simple_step_length: float = 2e-5,
    general_step_length: float = 4e-5,
    alpha_mode: AlphaMode = "negative-median",
    make_plots: bool = True,
    verbose: bool = False,
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
        general_max_iter=general_max_iter,
        step_max_iter=step_max_iter,
        replicate_max_iter=replicate_max_iter,
        replicate_count=replicate_count,
        simple_step_length=simple_step_length,
        general_step_length=general_step_length,
        alpha_mode=alpha_mode,
        make_plots=make_plots,
        verbose=verbose,
    )
    return run_mirna_simulation_suite(config)


def run_mirna_simulation_suite(config: SimulationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "run.log"

    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            return _run_mirna_simulation_suite(config, log_path=log_path)


def _run_mirna_simulation_suite(
    config: SimulationConfig,
    *,
    log_path: Path,
) -> dict[str, Any]:
    rd.seed(config.random_seed)
    np.random.seed(config.random_seed)

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
        step_length=config.simple_step_length,
        alpha_mode=config.alpha_mode,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
        verbose=config.verbose,
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
        verbose=config.verbose,
    )

    simple_replicate_result = run_simple_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=simple_result["logit_scores"],
        replicate_count=config.replicate_count,
        max_iter=config.replicate_max_iter,
        step_length=config.simple_step_length,
        stochastic_factor=config.stochastic_factor,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
        verbose=config.verbose,
    )

    general_matrix_result = run_general_matrix_experiment(
        mirna_sequences,
        gene_sequences,
        max_iter=config.general_max_iter,
        step_length=config.general_step_length,
        alpha_mode=config.alpha_mode,
        stochastic_factor=0.01,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
        verbose=config.verbose,
    )

    general_replicate_result = run_general_replicate_experiment(
        mirna_sequences,
        gene_sequences,
        logit_scores=general_matrix_result["logit_scores"],
        replicate_count=config.replicate_count,
        max_iter=config.replicate_max_iter,
        step_length=config.general_step_length,
        stochastic_factor=0.01,
        num_threads=config.num_threads,
        figures_dir=figures_dir,
        make_plots=config.make_plots,
        verbose=config.verbose,
    )

    summary = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "log_path": str(log_path),
        },
        "dataset": {
            "sequence_count": len(mirna_sequences),
            "dataset_index": config.dataset_index,
            "split": config.split,
        },
        "simple_model": summarize_simple_model(simple_result),
        "step_length_experiment": step_result,
        "simple_replicate_experiment": simple_replicate_result,
        "general_matrix_experiment": general_matrix_result,
        "general_replicate_experiment": general_replicate_result,
    }

    write_simulation_outputs(config.output_dir, summary)
    return summary
