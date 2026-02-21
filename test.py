"""Legacy entrypoint kept for compatibility.

Runs a short grokking experiment and prints a summary.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grokking import ExperimentConfig, run_experiment


if __name__ == "__main__":
    cfg = ExperimentConfig(
        modulus=97,
        train_fraction=0.3,
        hidden_sizes=(256, 256),
        learning_rate=0.03,
        weight_decay=0.02,
        steps=8_000,
        eval_every=100,
        output_dir="artifacts/test_py_demo",
    )
    summary = run_experiment(cfg)
    print(summary)
