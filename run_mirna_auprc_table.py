from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THREADS = 65
STEP_SCALES = "0.00001,0.00005,0.0001,0.0005"
MAX_ITERS = "100"
EVAL_SPLITS = "hejret_test,manakov_test,manakov_leftout"
RESULTS_DIR = Path("results/case_study_for_mirna")

RUNS = [
    ("hejret", "train_hejret_eval_all"),
    ("manakov", "train_manakov_eval_all"),
]


def run_case(train_family: str, run_tag: str) -> None:
    print(f"Training once on {train_family}_train; evaluating {EVAL_SPLITS}", flush=True)
    command = [
        sys.executable,
        "case_study_for_mirna/case_study_mirna.py",
        "--dataset",
        train_family,
        "--eval-splits",
        EVAL_SPLITS,
        "--aligner-modes",
        "local,global",
        "--substitution-modes",
        "simple,symmetric,general",
        "--gap-modes",
        "linear,affine",
        "--stepfunctions",
        "power,constant",
        "--step-scales",
        STEP_SCALES,
        "--max-iters",
        MAX_ITERS,
        "--num-threads",
        str(THREADS),
        "--results-dir",
        str(RESULTS_DIR),
        "--run-tag",
        run_tag,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for train_family, run_tag in RUNS:
        run_case(train_family, run_tag)
    print("All miRNA AUPRC table runs completed.", flush=True)


if __name__ == "__main__":
    main()
