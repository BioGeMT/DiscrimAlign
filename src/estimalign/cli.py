from __future__ import annotations

import argparse
from pathlib import Path

from estimalign.simulation_experiments import run_simulation_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="estimalign",
        description="EstimAlign simulation experiment workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run-simulation-experiments",
        help="Run converted simulation experiments.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simulation_experiments"),
        help="Directory for generated outputs.",
    )
    run_parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Index from miRBench list_datasets().",
    )
    run_parser.add_argument(
        "--split",
        default="train",
        help="miRBench split to use.",
    )
    run_parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    run_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on dataset rows for quick tests.",
    )
    run_parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of EstimAlign worker threads.",
    )
    run_parser.add_argument(
        "--simple-max-iter",
        type=int,
        default=50,
        help="Iterations for simple and general miRNA experiments.",
    )
    run_parser.add_argument(
        "--step-max-iter",
        type=int,
        default=10,
        help="Iterations for the step-length experiment.",
    )
    run_parser.add_argument(
        "--replicate-max-iter",
        type=int,
        default=5,
        help="Iterations for each replicate experiment.",
    )
    run_parser.add_argument(
        "--replicate-count",
        type=int,
        default=20,
        help="Number of replicate runs.",
    )
    run_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable figure generation.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-simulation-experiments":
        run_simulation_experiments(
            output_dir=args.output_dir,
            dataset_index=args.dataset_index,
            split=args.split,
            random_seed=args.random_seed,
            max_records=args.max_records,
            num_threads=args.num_threads,
            simple_max_iter=args.simple_max_iter,
            step_max_iter=args.step_max_iter,
            replicate_max_iter=args.replicate_max_iter,
            replicate_count=args.replicate_count,
            make_plots=not args.no_plots,
        )
        return 0

    parser.print_help()
    return 1