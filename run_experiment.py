from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grokking import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run modular arithmetic grokking experiments.")

    p.add_argument("--preset", choices=["grokking", "baseline"], default="grokking")
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--train-fraction", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=None, help="Hidden layer sizes, e.g. --hidden 256 256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Output directory under artifacts/")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.preset == "grokking":
        cfg = ExperimentConfig(
            modulus=args.modulus,
            train_fraction=args.train_fraction,
            hidden_sizes=(256, 256),
            learning_rate=0.12,
            weight_decay=0.001,
            steps=40_000,
            batch_size=512,
            eval_every=args.eval_every,
            seed=args.seed,
            output_dir="artifacts/grokking",
        )
    else:
        cfg = ExperimentConfig(
            modulus=args.modulus,
            train_fraction=args.train_fraction,
            hidden_sizes=(256, 256),
            learning_rate=0.12,
            weight_decay=0.0,
            steps=40_000,
            batch_size=512,
            eval_every=args.eval_every,
            seed=args.seed,
            output_dir="artifacts/baseline",
        )

    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.hidden is not None:
        cfg.hidden_sizes = tuple(args.hidden)
    if args.out is not None:
        cfg.output_dir = args.out

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    summary = run_experiment(cfg)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
