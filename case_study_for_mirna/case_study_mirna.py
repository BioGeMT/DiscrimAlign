from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
from Bio.Seq import Seq
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_study_for_mirna.import_mirbench_datasets import get_dataset_dataframe
from case_study_for_mirna.modeling import build_trajectory_rows, fit_configuration, rank_successful, summarize_result
from case_study_for_mirna.outputs import curve_point_rows, save_convergence_plot, save_pr_plot, save_roc_plot, write_rows
from case_study_for_mirna.scoring import evaluate_probabilities, score_pairs_with_model

PAIRED_DATASET_SPLITS = {"hejret": ("hejret_train", "hejret_test"), "manakov": ("manakov_train", "manakov_test")}
SUPPORTED_DATASET_SPLITS = ["hejret_train", "hejret_test", "manakov_train", "manakov_test", "manakov_leftout", "klimentova_test"]


def csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EstimAlign miRNA case-study grid.")
    parser.add_argument("--dataset", default="hejret", choices=sorted(PAIRED_DATASET_SPLITS))
    parser.add_argument("--dataset-split", default=None, choices=SUPPORTED_DATASET_SPLITS)
    parser.add_argument("--eval-splits", default=None)
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
    parser.add_argument("--final-max-iter", type=int, default=1000)
    parser.add_argument("--selection-objective", default="auto", choices=["auto", "validation_ap", "final_loglik"])
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--config-workers", type=int, default=1)
    parser.add_argument("--limit-configs", type=int, default=0)
    return parser.parse_args()


def prepare_inputs(frame):
    seq_a = frame["noncodingRNA"].astype(str).tolist()
    seq_b = [str(Seq(seq).reverse_complement()) for seq in frame["gene"].astype(str)]
    labels = frame["label"].astype(int).tolist()
    return seq_a, seq_b, labels


def load_frames(args):
    if args.dataset_split:
        frame = get_dataset_dataframe(args.dataset_split).reset_index(drop=True)
        return args.dataset_split, frame, frame, frame, {args.dataset_split: frame}
    train_alias, default_test_alias = PAIRED_DATASET_SPLITS[args.dataset]
    train_frame = get_dataset_dataframe(train_alias).reset_index(drop=True)
    split_names = csv_values(args.eval_splits) if args.eval_splits else [default_test_alias]
    invalid = [name for name in split_names if name not in SUPPORTED_DATASET_SPLITS]
    if invalid:
        raise ValueError(f"Invalid evaluation split aliases: {invalid}")
    evaluation_frames = {name: get_dataset_dataframe(name).reset_index(drop=True) for name in split_names}
    fit_frame, validation_frame = train_test_split(
        train_frame,
        test_size=args.validation_fraction,
        random_state=args.split_seed,
        stratify=train_frame["label"].astype(int),
    )
    return args.dataset, train_frame, fit_frame.reset_index(drop=True), validation_frame.reset_index(drop=True), evaluation_frames


def evaluate_model(config, result, inputs_by_split, run_dir):
    updates, metrics, pr_points, roc_points, split_stats = {}, [], [], [], {}
    for split_name, inputs in inputs_by_split.items():
        print(f"  Scoring split {split_name} ({len(inputs[2])} pairs)", flush=True)
        probabilities = score_pairs_with_model(
            inputs[0],
            inputs[1],
            result["aligner"],
            result["alpha"],
            num_threads=int(config["num_threads"]),
        )
        stats = evaluate_probabilities(inputs[2], probabilities)
        split_stats[split_name] = stats
        updates[f"ap_{split_name}"] = stats["average_precision"]
        updates[f"roc_auc_{split_name}"] = stats["roc_auc"]
        metrics.append({**config, "split": split_name, "average_precision": stats["average_precision"], "roc_auc": stats["roc_auc"], "status": "ok"})
        pr_points.extend(curve_point_rows(config["config"], split_name, stats, "pr"))
        roc_points.extend(curve_point_rows(config["config"], split_name, stats, "roc"))
    reference_split = "fit" if "fit" in split_stats else "train"
    if reference_split in split_stats:
        for split_name, stats in split_stats.items():
            if split_name == reference_split:
                continue
            save_pr_plot(
                split_stats[reference_split],
                stats,
                run_dir / "pr_curves" / f"{config['config']}_{reference_split}_vs_{split_name}.png",
                f"{config['config']}: {reference_split} vs {split_name}",
                train_label=reference_split,
                test_label=split_name,
            )
            save_roc_plot(
                split_stats[reference_split],
                stats,
                run_dir / "roc_curves" / f"{config['config']}_{reference_split}_vs_{split_name}.png",
                f"{config['config']}: {reference_split} vs {split_name}",
                train_label=reference_split,
                test_label=split_name,
            )
    return updates, metrics, pr_points, roc_points


def build_config(dataset_label, index, values, num_threads):
    aligner_mode, gap_mode, substitution_mode, stepfunction, step_scale, max_iter = values
    config_name = f"cfg_{index:04d}_{aligner_mode}_{gap_mode}_{substitution_mode}_{stepfunction}_s{step_scale}_i{max_iter}"
    return {
        "dataset": dataset_label,
        "config_index": index,
        "config": config_name,
        "aligner_mode": aligner_mode,
        "gap_mode": gap_mode,
        "substitution_mode": substitution_mode,
        "stepfunction": stepfunction,
        "step_scale": step_scale,
        "max_iter": max_iter,
        "num_threads": num_threads,
    }


def run_configuration(index, total_configs, config, fit_inputs, inputs_by_split, run_dir):
    print(f"Starting {index}/{total_configs}: {config['config']}", flush=True)
    try:
        result, runtime = fit_configuration(fit_inputs, config)
        row = summarize_result(config, result, runtime)
        updates, metrics, pr_points, roc_points = evaluate_model(config, result, inputs_by_split, run_dir)
        row.update(updates)
        trajectories = build_trajectory_rows(config, result)
        save_convergence_plot(trajectories, run_dir / "convergence" / f"{config['config']}.png", config["config"])
        print(f"Completed {index}/{total_configs}: ok", flush=True)
        return {
            "summary": row,
            "errors": [],
            "metrics": metrics,
            "pr_points": pr_points,
            "roc_points": roc_points,
            "trajectories": trajectories,
        }
    except Exception as exc:
        row = {**config, "status": "error", "error": repr(exc), "final_loglik": np.nan}
        print(f"Completed {index}/{total_configs}: error", flush=True)
        print(f"  {row['error']}", flush=True)
        return {
            "summary": row,
            "errors": [row],
            "metrics": [],
            "pr_points": [],
            "roc_points": [],
            "trajectories": [],
        }


def main():
    args = parse_args()
    dataset_label, train_frame, fit_frame, validation_frame, evaluation_frames = load_frames(args)
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = Path(args.results_dir) / f"{dataset_label}{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_inputs = prepare_inputs(train_frame)
    fit_inputs = prepare_inputs(fit_frame)
    validation_inputs = prepare_inputs(validation_frame)
    evaluation_inputs = {name: prepare_inputs(frame) for name, frame in evaluation_frames.items()}
    inputs_by_split = {"fit": fit_inputs, "validation": validation_inputs, **evaluation_inputs}
    grid = list(product(csv_values(args.aligner_modes), csv_values(args.gap_modes), csv_values(args.substitution_modes), csv_values(args.stepfunctions), [float(v) for v in csv_values(args.step_scales)], [int(v) for v in csv_values(args.max_iters)]))
    if args.limit_configs:
        grid = grid[: args.limit_configs]
    summary_rows, error_rows, metric_rows, pr_rows, roc_rows, trajectory_rows = [], [], [], [], [], []
    print(f"Running {len(grid)} configurations...", flush=True)
    configs = [build_config(dataset_label, index, values, args.num_threads) for index, values in enumerate(grid, start=1)]
    if args.config_workers == 1:
        results = [
            run_configuration(index, len(configs), config, fit_inputs, inputs_by_split, run_dir)
            for index, config in enumerate(configs, start=1)
        ]
    else:
        results = Parallel(n_jobs=args.config_workers, return_as="list")(
            delayed(run_configuration)(index, len(configs), config, fit_inputs, inputs_by_split, run_dir)
            for index, config in enumerate(configs, start=1)
        )
    for result in results:
        summary_rows.append(result["summary"])
        error_rows.extend(result["errors"])
        metric_rows.extend(result["metrics"])
        pr_rows.extend(result["pr_points"])
        roc_rows.extend(result["roc_points"])
        trajectory_rows.extend(result["trajectories"])
    objective = "final_loglik" if args.selection_objective == "auto" and args.dataset_split else args.selection_objective.replace("validation_ap", "validation_ap")
    ranked = rank_successful(summary_rows, objective)
    if ranked and args.final_max_iter > 0:
        best = ranked[0]
        final_config = {key: best[key] for key in ["dataset", "aligner_mode", "gap_mode", "substitution_mode", "stepfunction", "step_scale", "num_threads"]}
        final_config.update({"config_index": 0, "config": f"final_refit_{best['config']}", "max_iter": args.final_max_iter})
        result, runtime = fit_configuration(train_inputs, final_config)
        final_row = summarize_result(final_config, result, runtime)
        final_updates, final_metrics, final_pr, final_roc = evaluate_model(final_config, result, {"train": train_inputs, **evaluation_inputs}, run_dir / "final_refit")
        final_row.update(final_updates)
        final_trajectories = build_trajectory_rows(final_config, result)
        write_rows(run_dir / "final_refit" / "summary.csv", [final_row])
        write_rows(run_dir / "final_refit" / "metrics.csv", final_metrics)
        write_rows(run_dir / "final_refit" / "pr_points.csv", final_pr)
        write_rows(run_dir / "final_refit" / "roc_points.csv", final_roc)
        write_rows(run_dir / "final_refit" / "trajectory.csv", final_trajectories)
        save_convergence_plot(final_trajectories, run_dir / "final_refit" / "convergence.png", final_config["config"])
    write_rows(run_dir / "summary.csv", summary_rows)
    write_rows(run_dir / "errors.csv", error_rows)
    write_rows(run_dir / "metrics.csv", metric_rows)
    write_rows(run_dir / "pr_points.csv", pr_rows)
    write_rows(run_dir / "roc_points.csv", roc_rows)
    write_rows(run_dir / "trajectories.csv", trajectory_rows)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
