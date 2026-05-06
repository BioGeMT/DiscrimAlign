from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

FIGURE_DPI = 300

plt.rcParams.update(
    {
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_histogram(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    plt.figure(figsize=(5.8, 3.8))
    plt.hist(values, bins=60, edgecolor="black", linewidth=0.3)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    save_figure(output_path)


def plot_label_scatter(
    x_values: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    plt.figure(figsize=(5.8, 3.8))
    plt.plot(x_values, labels, ".", alpha=0.2, markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel("Simulated label")
    plt.yticks([0, 1], ["0", "1"])
    plt.title(title)
    save_figure(output_path)


def plot_optimization_trajectories(
    params: dict[str, Any],
    *,
    max_iter: int,
    true_loglik: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))

    axes[0, 0].plot(np.arange(max_iter), params["subgradient_l2_trajectory"], linewidth=1.5)
    axes[0, 0].axhline(0, linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Subgradient norm")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("L2 norm")

    axes[0, 1].plot(np.arange(max_iter + 1), params["loglik_trajectory"], linewidth=1.5)
    axes[0, 1].axhline(true_loglik, linestyle="--", linewidth=1.0, label="Simulation truth")
    axes[0, 1].set_title("Log-likelihood")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Log-likelihood")
    axes[0, 1].legend(frameon=False)

    start = max_iter // 2
    axes[1, 0].plot(
        np.arange(start, max_iter),
        params["subgradient_l2_trajectory"][start:],
        linewidth=1.5,
    )
    axes[1, 0].axhline(0, linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Subgradient norm, second half")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("L2 norm")

    axes[1, 1].plot(
        np.arange(start, max_iter + 1),
        params["loglik_trajectory"][start:],
        linewidth=1.5,
    )
    axes[1, 1].axhline(true_loglik, linestyle="--", linewidth=1.0, label="Simulation truth")
    axes[1, 1].set_title("Log-likelihood, second half")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Log-likelihood")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Optimization trajectory", y=1.02)
    save_figure(output_path)


def plot_step_length_loglikelihoods(
    results: list[dict[str, Any]],
    step_lengths: np.ndarray,
    *,
    max_iter: int,
    true_loglik: float,
    output_path: Path,
    xlim: tuple[int, int] | None = None,
) -> None:
    plt.figure(figsize=(6.4, 4.2))
    for params, step_length in zip(results, step_lengths):
        plt.plot(
            np.arange(max_iter + 1),
            params["loglik_trajectory"],
            alpha=0.8,
            linewidth=1.2,
            label=f"{step_length:.1e}",
        )

    plt.axhline(true_loglik, linestyle="--", linewidth=1.0, label="Simulation truth")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Step-length comparison")
    plt.legend(title="Step length", frameon=False, ncol=2)

    if xlim is not None:
        plt.xlim(*xlim)

    save_figure(output_path)


def plot_replicate_loglikelihoods(
    results: list[dict[str, Any]],
    true_logliks: list[float],
    *,
    max_iter: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    for params in results:
        loglik_trajectory = np.asarray(params["loglik_trajectory"], dtype=float)
        axes[0].plot(
            np.arange(max_iter + 1),
            loglik_trajectory,
            alpha=0.35,
            linewidth=1.2,
        )
    axes[0].set_title("Replicate log-likelihoods")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Log-likelihood")

    axes[1].axhline(0, linestyle="--", linewidth=1.0)
    for params, true_loglik in zip(results, true_logliks):
        loglik_trajectory = np.asarray(params["loglik_trajectory"], dtype=float)
        axes[1].plot(
            np.arange(max_iter + 1),
            true_loglik - loglik_trajectory,
            alpha=0.35,
            linewidth=1.2,
        )
    axes[1].set_title("Truth minus fitted log-likelihood")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Difference")

    fig.suptitle("Replicate optimization behavior", y=1.04)
    save_figure(output_path)


def plot_substitution_comparison(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    rows = comparison["rows"]
    true_values = np.array([row["true"] for row in rows])
    estimated_values = np.array([row["estimated"] for row in rows])

    lower = float(min(true_values.min(), estimated_values.min()))
    upper = float(max(true_values.max(), estimated_values.max()))

    plt.figure(figsize=(4.8, 4.6))
    plt.plot(true_values, estimated_values, ".", markersize=6, alpha=0.8)
    plt.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, label="Identity")
    plt.xlabel("True substitution score")
    plt.ylabel("Estimated substitution score")
    plt.title("Substitution matrix recovery")
    plt.legend(frameon=False)
    plt.axis("equal")
    save_figure(output_path)
