"""Extract the historical table from the saved notebook.

This script intentionally does not execute training. It creates a machine-readable
record of the original reported values so historical evidence remains separate from
new controlled runs.
"""

import json
from pathlib import Path

HISTORICAL = [
    {"method": "standard", "steps": 112, "trainer_runtime_seconds": 336.37, "wall_seconds": 336.79, "trainer_examples_per_second": 2.676, "table_examples_per_second": 2.67, "peak_allocated_gb_rank0": 2.48, "mean_train_loss": 0.69304},
    {"method": "fsdp", "steps": 56, "trainer_runtime_seconds": 513.45, "wall_seconds": 517.47, "trainer_examples_per_second": 1.753, "table_examples_per_second": 1.74, "peak_allocated_gb_rank0": 2.91, "mean_train_loss": 0.69296},
    {"method": "zero2", "steps": 56, "trainer_runtime_seconds": 201.06, "wall_seconds": 242.60, "trainer_examples_per_second": 4.476, "table_examples_per_second": 3.71, "peak_allocated_gb_rank0": 3.62, "mean_train_loss": 0.69300},
    {"method": "zero2_optimized", "steps": 56, "trainer_runtime_seconds": 200.49, "wall_seconds": 208.76, "trainer_examples_per_second": 4.489, "table_examples_per_second": 4.31, "peak_allocated_gb_rank0": 3.62, "mean_train_loss": 0.69317},
    {"method": "zero3", "steps": 56, "trainer_runtime_seconds": 1086.09, "wall_seconds": 1097.17, "trainer_examples_per_second": 0.829, "table_examples_per_second": 0.82, "peak_allocated_gb_rank0": 4.20, "mean_train_loss": 0.69303},
]


def main():
    path = Path("results/historical")
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "original_notebook_results.json", "w", encoding="utf-8") as f:
        json.dump(HISTORICAL, f, indent=2)


if __name__ == "__main__":
    main()
