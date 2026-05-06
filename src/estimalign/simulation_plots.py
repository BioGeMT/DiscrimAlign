from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt


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
