from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.config import (
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    TaskConfig,
    TrainingConfig,
)
from grokking.experiment import run_experiment


class ExperimentSmokeTests(unittest.TestCase):
    def test_smoke_run_writes_artifacts(self) -> None:
        cfg = ExperimentConfig(
            name="smoke",
            task=TaskConfig(kind="modular", operation="add", modulus=13),
            model=ModelConfig(hidden_dims=[32], activation="relu"),
            optimizer=OptimizerConfig(name="adamw", lr=0.01, weight_decay=0.0),
            training=TrainingConfig(
                steps=30,
                batch_size=64,
                train_fraction=0.5,
                seed=1,
                eval_every=10,
                metric_threshold=0.95,
            ),
            logging=LoggingConfig(verbose=False, save_curves=False),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = run_experiment(cfg, out)

            self.assertTrue((out / "metrics.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "config.json").exists())

            self.assertIn("final_train_acc", summary)
            self.assertIn("final_val_acc", summary)
            self.assertEqual(summary["steps"], 30)


if __name__ == "__main__":
    unittest.main()
