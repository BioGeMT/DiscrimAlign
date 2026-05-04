from __future__ import annotations

from pathlib import Path


def run_simulation_experiments(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / "README.txt"
    marker.write_text(
        "Simulation experiment workflow initialized.\n",
        encoding="utf-8",
    )