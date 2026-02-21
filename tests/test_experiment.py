from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grokking.experiment import build_modular_addition_data, split_train_val  # noqa: E402


class ExperimentTests(unittest.TestCase):
    def test_modular_addition_labels_are_correct(self) -> None:
        modulus = 11
        x, y, labels = build_modular_addition_data(modulus)

        self.assertEqual(x.shape, (modulus * modulus, 2 * modulus))
        self.assertEqual(y.shape, (modulus * modulus, modulus))
        self.assertEqual(labels.shape, (modulus * modulus,))

        # Example: 7 + 9 mod 11 = 5
        idx = 7 * modulus + 9
        self.assertEqual(int(labels[idx]), 5)

    def test_split_sizes_match_fraction(self) -> None:
        rng = np.random.default_rng(0)
        x = np.zeros((100, 4), dtype=np.float32)
        y = np.zeros((100, 2), dtype=np.float32)

        x_train, y_train, x_val, y_val = split_train_val(x, y, train_fraction=0.3, rng=rng)
        self.assertEqual(x_train.shape[0], 30)
        self.assertEqual(y_train.shape[0], 30)
        self.assertEqual(x_val.shape[0], 70)
        self.assertEqual(y_val.shape[0], 70)


if __name__ == "__main__":
    unittest.main()
