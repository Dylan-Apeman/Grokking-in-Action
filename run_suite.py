#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grokking.config import list_presets
from grokking.sweeps import run_seed_sweep


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preset sweeps for grokking experiments.")
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(list_presets()),
        help="Comma-separated preset names",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps for all runs")
    parser.add_argument("--out", type=str, default="artifacts/sweeps", help="Output root")
    parser.add_argument("--no-curves", action="store_true", help="Disable plot generation")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    args = parser.parse_args()

    presets = _parse_csv_strs(args.presets)
    seeds = _parse_csv_ints(args.seeds)

    rows = run_seed_sweep(
        presets=presets,
        seeds=seeds,
        out_root=args.out,
        steps_override=args.steps,
        save_curves_override=False if args.no_curves else None,
        verbose=not args.quiet,
    )

    print(json.dumps({"num_runs": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
