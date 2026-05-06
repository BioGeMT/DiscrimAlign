from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    output_dir: Path
    dataset_index: int = 0
    split: str = "train"
    random_seed: int = 7
    max_records: int | None = None
    num_threads: int = 16
    stochastic_factor: float = 0.001
    simple_max_iter: int = 50
    step_max_iter: int = 10
    replicate_max_iter: int = 5
    replicate_count: int = 20
    make_plots: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class SimpleModelTruth:
    match: float = 1.0
    mismatch: float = -1.0
    gap_open: float = -1.2
    gap_extend: float = -0.1
    alpha: float = -9.0


@dataclass(frozen=True)
class GeneralMatrixTruth:
    gap_open: float = -1.2
    gap_extend: float = -0.1
    alpha: float = -12.0
