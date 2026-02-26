from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import get_preset_config
from .experiment import run_experiment


def run_seed_sweep(
    presets: list[str],
    seeds: list[int],
    out_root: str | Path,
    steps_override: int | None = None,
    save_curves_override: bool | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for preset in presets:
        for seed in seeds:
            cfg = get_preset_config(preset)
            cfg.training.seed = seed
            if steps_override is not None:
                cfg.training.steps = steps_override
            if save_curves_override is not None:
                cfg.logging.save_curves = save_curves_override
            cfg.name = f"{preset}_seed{seed}"

            run_dir = out_root / preset / f"seed_{seed}"
            if verbose:
                print(f"\n=== Running {preset} (seed={seed}) -> {run_dir} ===")

            summary = run_experiment(cfg, run_dir)
            row = {
                "preset": preset,
                "seed": seed,
                "final_train_acc": summary["final_train_acc"],
                "final_val_acc": summary["final_val_acc"],
                "memorization_step": summary["memorization_step"],
                "generalization_step": summary["generalization_step"],
                "runtime_sec": summary["runtime_sec"],
                "run_dir": str(run_dir),
            }
            rows.append(row)

    csv_path = out_root / "sweep_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(out_root / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return rows
