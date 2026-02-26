from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from .config import TaskConfig


@dataclass
class TaskDataset:
    pairs: np.ndarray  # [N, 2], token ids
    labels: np.ndarray  # [N], class ids
    vocab_size: int
    num_classes: int
    metadata: dict[str, Any]


@dataclass
class SplitDataset:
    train_pairs: np.ndarray
    train_labels: np.ndarray
    val_pairs: np.ndarray
    val_labels: np.ndarray


def _build_modular_dataset(config: TaskConfig) -> TaskDataset:
    if config.modulus <= 1:
        raise ValueError("modulus must be > 1")

    m = config.modulus
    op = config.operation.lower()

    x, y = np.indices((m, m))
    pairs = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.int64)

    a = pairs[:, 0]
    b = pairs[:, 1]
    if op == "add":
        labels = (a + b) % m
    elif op == "sub":
        labels = (a - b) % m
    elif op == "mul":
        labels = (a * b) % m
    else:
        raise ValueError(f"Unsupported modular operation: {config.operation}")

    return TaskDataset(
        pairs=pairs,
        labels=labels.astype(np.int64),
        vocab_size=m,
        num_classes=m,
        metadata={"task": "modular", "operation": op, "modulus": m},
    )


def _compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    # Group composition p ∘ q: first q, then p.
    return tuple(p[q[i]] for i in range(len(p)))


def _build_permutation_dataset(config: TaskConfig) -> TaskDataset:
    n = config.permutation_size
    if n < 2 or n > 7:
        raise ValueError("permutation_size must be in [2, 7]")

    elems = list(permutations(range(n)))
    index_of = {perm: idx for idx, perm in enumerate(elems)}
    group_size = len(elems)

    x, y = np.indices((group_size, group_size))
    pairs = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.int64)

    labels = np.empty(pairs.shape[0], dtype=np.int64)
    for i, (a_idx, b_idx) in enumerate(pairs):
        composed = _compose(elems[a_idx], elems[b_idx])
        labels[i] = index_of[composed]

    return TaskDataset(
        pairs=pairs,
        labels=labels,
        vocab_size=group_size,
        num_classes=group_size,
        metadata={"task": "permutation", "n": n, "group_size": group_size},
    )


def build_dataset(config: TaskConfig) -> TaskDataset:
    kind = config.kind.lower()
    if kind == "modular":
        return _build_modular_dataset(config)
    if kind == "permutation":
        return _build_permutation_dataset(config)
    raise ValueError(f"Unsupported task kind: {config.kind}")


def split_dataset(
    pairs: np.ndarray,
    labels: np.ndarray,
    train_fraction: float,
    seed: int,
) -> SplitDataset:
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")

    n = pairs.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_size = int(n * train_fraction)
    train_size = min(max(train_size, 1), n)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return SplitDataset(
        train_pairs=pairs[train_idx],
        train_labels=labels[train_idx],
        val_pairs=pairs[val_idx],
        val_labels=labels[val_idx],
    )


def encode_pairs_one_hot(pairs: np.ndarray, vocab_size: int) -> np.ndarray:
    n = pairs.shape[0]
    out = np.zeros((n, vocab_size * 2), dtype=np.float32)

    rows = np.arange(n)
    out[rows, pairs[:, 0]] = 1.0
    out[rows, vocab_size + pairs[:, 1]] = 1.0
    return out
