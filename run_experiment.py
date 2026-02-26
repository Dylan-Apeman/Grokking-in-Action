#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grokking.config import apply_overrides, get_preset_config, list_presets, load_config_file
from grokking.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one grokking experiment.")
    parser.add_argument("--preset", type=str, default="mod_add_baseline", help="Preset config name.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument("--override", action="append", default=[], help="Override like training.steps=20000")
    parser.add_argument("--steps", type=int, default=None, help="Shortcut override for training.steps")
    parser.add_argument("--out", type=str, default="artifacts/run", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Disable step-wise logging")
    parser.add_argument("--no-curves", action="store_true", help="Disable plot generation")
    parser.add_argument("--list-presets", action="store_true", help="List built-in presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name in list_presets():
            print(name)
        return

    if args.config:
        cfg = load_config_file(args.config)
    else:
        cfg = get_preset_config(args.preset)

    if args.steps is not None:
        cfg.training.steps = args.steps

    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if args.quiet:
        cfg.logging.verbose = False
    if args.no_curves:
        cfg.logging.save_curves = False

    summary = run_experiment(cfg, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
