from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.config import TaskConfig
from grokking.tasks import build_dataset, split_dataset


class TaskDatasetTests(unittest.TestCase):
    def test_modular_add_labels(self) -> None:
        cfg = TaskConfig(kind="modular", operation="add", modulus=7)
        ds = build_dataset(cfg)

        self.assertEqual(ds.pairs.shape, (49, 2))
        self.assertEqual(ds.num_classes, 7)

        expected = (ds.pairs[:, 0] + ds.pairs[:, 1]) % 7
        self.assertTrue(np.array_equal(ds.labels, expected))

    def test_permutation_closure(self) -> None:
        cfg = TaskConfig(kind="permutation", permutation_size=3)
        ds = build_dataset(cfg)

        perms = list(permutations(range(3)))
        index_of = {p: i for i, p in enumerate(perms)}

        self.assertEqual(ds.vocab_size, 6)
        self.assertEqual(ds.pairs.shape[0], 36)

        for a_idx, b_idx, label in zip(ds.pairs[:18, 0], ds.pairs[:18, 1], ds.labels[:18]):
            p = perms[int(a_idx)]
            q = perms[int(b_idx)]
            composed = tuple(p[q[i]] for i in range(3))
            self.assertEqual(label, index_of[composed])

    def test_split_is_deterministic(self) -> None:
        cfg = TaskConfig(kind="modular", operation="mul", modulus=11)
        ds = build_dataset(cfg)

        s1 = split_dataset(ds.pairs, ds.labels, train_fraction=0.3, seed=42)
        s2 = split_dataset(ds.pairs, ds.labels, train_fraction=0.3, seed=42)

        self.assertTrue(np.array_equal(s1.train_pairs, s2.train_pairs))
        self.assertTrue(np.array_equal(s1.train_labels, s2.train_labels))
        self.assertEqual(s1.train_pairs.shape[0], int(121 * 0.3))


if __name__ == "__main__":
    unittest.main()
