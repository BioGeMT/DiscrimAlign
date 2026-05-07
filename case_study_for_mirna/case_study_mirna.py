from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.estimalign import estimalign
from src.optimization import create_constant_step, create_powerstep
from case_study_for_mirna.import_mirbench_datasets import get_dataset_dataframe

PAIRED_DATASET_SPLITS = {
    "hejret": ("hejret_train", "hejret_test"),
    "manakov": ("manakov_train", "manakov_test"),
}
SUPPORTED_DATASET_SPLITS = [
    "hejret_train",
    "hejret_test",
    "manakov_train",
    "manakov_test",
    "manakov_leftout",
    "klimentova_test",
]


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(value) for value in _parse_csv(raw)]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(value) for value in _parse_csv(raw)]


def _parse_dataset_splits(raw: str | None) -> list[str]:
    if not raw:
        return []
    aliases = _parse_csv(raw)
    invalid = [alias for alias in aliases if alias not in SUPPORTED_DATASET_SPLITS]
    if invalid:
        valid = ", ".join(SUPPORTED_DATASET_SPLITS)
        raise ValueError(f"Invalid dataset split aliases: {invalid}. Valid aliases: {valid}")
    return aliases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EstimAlign miRNA case-study calculations with datasets obtained through miRBench."
    )
    parser.add_argument("--dataset", default="hejret", choices=sorted(PAIRED_DATASET_SPLITS))
    parser.add_argument("--dataset-split", default=None, choices=SUPPORTED_DATASET_SPLITS)
    parser.add_argument(
        "--eval-splits",
        default=None,
        help=(
            "Comma-separated dataset split aliases evaluated after training once on "
            "<dataset>_train, for example: hejret_test,manakov_test,manakov_leftout."
        ),
    )
    parser.add_argument("--results-dir", default="results/case_study_for_mirna")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--aligner-modes", default="local,global")
    parser.add_argument("--gap-modes", default="affine,linear")
    parser.add_argument("--substitution-modes", default="simple,symmetric,general")
    parser.add_argument("--stepfunctions", default="power,constant")
    parser.add_argument("--step-scales", default="0.00001,0.00005,0.0001,0.0005,0.001")
    parser.add_argument("--max-iters", default="100,200")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--limit-configs", type=int, default=0)
    return parser.parse_args()


def load_dataset(alias: str) -> pd.DataFrame:
    return get_dataset_dataframe(alias)


def prepare_inputs(df: pd.DataFrame):
    seqlist_a = df["noncodingRNA"].astype(str).tolist()
    seqlist_b = [str(Seq(seq).reverse_complement()) for seq in df["gene"].astype(str)]
    labels = df["label"].astype(int).tolist()
    return seqlist_a, seqlist_b, labels


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def score_pairs(seqlist_a, seqlist_b, aligner, alpha):
    scores = [next(aligner.align(seq_a, seq_b)).score for seq_a, seq_b in zip(seqlist_a, seqlist_b)]
    return sigmoid(np.asarray(scores, dtype=float) + float(alpha))


def make_stepfunction(name: str, scale: float):
    if name == "power":
        return create_powerstep(scale=scale)
    if name == "constant":
        return create_constant_step(scale=scale)
    raise ValueError(f"Unknown stepfunction: {name}")


def evaluate(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    metrics = {"average_precision": np.nan, "roc_auc": np.nan}
    if probabilities.size and np.isfinite(probabilities).all():
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = _parse_dataset_splits(args.eval_splits)

    if args.dataset_split and eval_splits:
        raise ValueError("--dataset-split and --eval-splits are mutually exclusive.")

    if args.dataset_split:
        train_df = load_dataset(args.dataset_split)
        fit_df = train_df
        val_df = train_df
        eval_frames = {args.dataset_split: train_df}
        dataset_label = args.dataset_split
    else:
        train_alias, default_eval_alias = PAIRED_DATASET_SPLITS[args.dataset]
        train_df = load_dataset(train_alias)
        selected_eval_splits = eval_splits or [default_eval_alias]
        eval_frames = {alias: load_dataset(alias) for alias in selected_eval_splits}
        fit_df, val_df = train_test_split(
            train_df,
            test_size=args.validation_fraction,
            random_state=args.split_seed,
            stratify=train_df["label"].astype(int),
        )
        dataset_label = args.dataset

    fit_inputs = prepare_inputs(fit_df)
    val_inputs = prepare_inputs(val_df)
    eval_inputs = {alias: prepare_inputs(frame) for alias, frame in eval_frames.items()}

    grid = list(
        product(
            _parse_csv(args.aligner_modes),
            _parse_csv(args.gap_modes),
            _parse_csv(args.substitution_modes),
            _parse_csv(args.stepfunctions),
            _parse_float_csv(args.step_scales),
            _parse_int_csv(args.max_iters),
        )
    )
    if args.limit_configs:
        grid = grid[: args.limit_configs]

    rows = []
    for idx, (aligner_mode, gap_mode, substitution_mode, step_name, step_scale, max_iter) in enumerate(grid, start=1):
        config = {
            "dataset": dataset_label,
            "config_index": idx,
            "aligner_mode": aligner_mode,
            "gap_mode": gap_mode,
            "substitution_mode": substitution_mode,
            "stepfunction": step_name,
            "step_scale": step_scale,
            "max_iter": max_iter,
            "num_threads": args.num_threads,
        }
        try:
            result = estimalign(
                seqlistA=fit_inputs[0],
                seqlistB=fit_inputs[1],
                labels=fit_inputs[2],
                aligner_mode=aligner_mode,
                gap_mode=gap_mode,
                substitution_mode=substitution_mode,
                stepfunction=make_stepfunction(step_name, step_scale),
                max_iter=max_iter,
                num_threads=args.num_threads,
                verbose=False,
            )
            val_prob = score_pairs(val_inputs[0], val_inputs[1], result["aligner"], result["alpha"])
            row = {**config, "status": "ok", "final_loglik": result.get("final_loglik", np.nan)}
            row.update({f"validation_{k}": v for k, v in evaluate(val_inputs[2], val_prob).items()})
            for alias, inputs in eval_inputs.items():
                eval_prob = score_pairs(inputs[0], inputs[1], result["aligner"], result["alpha"])
                row.update({f"{alias}_{k}": v for k, v in evaluate(inputs[2], eval_prob).items()})
        except Exception as exc:
            row = {**config, "status": "error", "error": repr(exc), "final_loglik": np.nan}
        rows.append(row)
        print(f"Completed {idx}/{len(grid)}: {row['status']}", flush=True)

    tag = f"_{args.run_tag}" if args.run_tag else ""
    summary_path = results_dir / f"{dataset_label}{tag}_summary.csv"
    columns = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
