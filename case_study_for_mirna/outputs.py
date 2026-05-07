from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def curve_point_rows(config_name: str, split_name: str, evaluation: dict, kind: str) -> list[dict]:
    rows = []
    if kind == "pr":
        for index, (recall, precision) in enumerate(zip(evaluation["recall"], evaluation["precision"])):
            rows.append({"config": config_name, "split": split_name, "kind": kind, "point_index": index, "recall": recall, "precision": precision})
    elif kind == "roc":
        for index, (fpr, tpr) in enumerate(zip(evaluation["fpr"], evaluation["tpr"])):
            rows.append({"config": config_name, "split": split_name, "kind": kind, "point_index": index, "fpr": fpr, "tpr": tpr})
    return rows


def save_xy_plot(x_values, y_values, out_path: str | Path, xlabel: str, ylabel: str, title: str) -> bool:
    if plt is None or len(x_values) == 0 or len(y_values) == 0:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def save_convergence_plot(rows: list[dict], out_path: str | Path, title: str) -> bool:
    if plt is None or not rows:
        return False
    loglik_rows = [row for row in rows if np.isfinite(row.get("loglik", np.nan))]
    subgrad_rows = [row for row in rows if np.isfinite(row.get("subgradient_l2", np.nan))]
    if not loglik_rows and not subgrad_rows:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    if loglik_rows:
        axes[0].plot([row["iteration"] for row in loglik_rows], [row["loglik"] for row in loglik_rows], linewidth=2)
    if subgrad_rows:
        axes[1].plot([row["iteration"] for row in subgrad_rows], [row["subgradient_l2"] for row in subgrad_rows], linewidth=2)
    axes[0].set_title(title)
    axes[0].set_ylabel("Log-likelihood")
    axes[1].set_ylabel("Subgradient L2 norm")
    axes[1].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True
