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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-simulation-experiments":
        run_simulation_experiments(output_dir=args.output_dir)
        return 0

    parser.print_help()
    return 1