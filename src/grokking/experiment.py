from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None


@dataclass
class ExperimentConfig:
    # Task
    modulus: int = 97
    train_fraction: float = 0.3
    seed: int = 0

    # Model
    hidden_sizes: tuple[int, ...] = (256, 256)

    # Optimization
    learning_rate: float = 0.03
    weight_decay: float = 0.02
    steps: int = 30_000
    batch_size: int = 512
    eval_every: int = 100

    # Output
    output_dir: str = "artifacts/default"


class MLP:
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int, rng: np.random.Generator):
        layer_sizes = [input_dim, *hidden_sizes, output_dim]
        self.ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / in_dim)
            w = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            self.ws.append(w)
            self.bs.append(b)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        zs: list[np.ndarray] = []
        hs: list[np.ndarray] = [x]
        for i in range(len(self.ws) - 1):
            z = hs[-1] @ self.ws[i] + self.bs[i]
            h = self.relu(z)
            zs.append(z)
            hs.append(h)
        logits = hs[-1] @ self.ws[-1] + self.bs[-1]
        probs = self.softmax(logits)
        return zs, hs, logits, probs

    def step(self, x: np.ndarray, y: np.ndarray, lr: float, weight_decay: float) -> float:
        zs, hs, _, probs = self.forward(x)
        batch_size = x.shape[0]

        eps = 1e-9
        data_loss = -np.mean(np.sum(y * np.log(probs + eps), axis=1))
        l2_loss = 0.5 * sum(np.sum(w * w) for w in self.ws)
        total_loss = float(data_loss + weight_decay * l2_loss)

        dlogits = (probs - y) / batch_size
        dws: list[np.ndarray] = [np.zeros_like(w) for w in self.ws]
        dbs: list[np.ndarray] = [np.zeros_like(b) for b in self.bs]

        dws[-1] = hs[-1].T @ dlogits + weight_decay * self.ws[-1]
        dbs[-1] = np.sum(dlogits, axis=0, keepdims=True)

        dh = dlogits @ self.ws[-1].T
        for i in range(len(self.ws) - 2, -1, -1):
            dz = dh * self.relu_grad(zs[i])
            dws[i] = hs[i].T @ dz + weight_decay * self.ws[i]
            dbs[i] = np.sum(dz, axis=0, keepdims=True)
            dh = dz @ self.ws[i].T

        for i in range(len(self.ws)):
            self.ws[i] -= lr * dws[i]
            self.bs[i] -= lr * dbs[i]

        return total_loss


def one_hot(indices: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], classes), dtype=np.float32)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def build_modular_addition_data(modulus: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(modulus), np.arange(modulus), indexing="ij")
    x_flat = xs.reshape(-1)
    y_flat = ys.reshape(-1)
    labels = (x_flat + y_flat) % modulus

    x_oh = one_hot(x_flat, modulus)
    y_oh = one_hot(y_flat, modulus)
    features = np.concatenate([x_oh, y_oh], axis=1)
    targets = one_hot(labels, modulus)
    return features, targets, labels


def accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == np.argmax(targets, axis=1)))


def split_train_val(x: np.ndarray, y: np.ndarray, train_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    n = x.shape[0]
    idx = rng.permutation(n)
    n_train = max(1, int(n * train_fraction))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path: Path, metrics: list[dict[str, float]], cfg: ExperimentConfig) -> None:
    if plt is None:
        return

    steps = [m["step"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    train_acc = [m["train_acc"] for m in metrics]
    val_acc = [m["val_acc"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(steps, train_loss, label="train loss", color="#c0392b", linewidth=2)
    axes[0].plot(steps, val_loss, label="val loss", color="#2980b9", linewidth=2)
    axes[0].set_title("Cross-Entropy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, train_acc, label="train acc", color="#16a085", linewidth=2)
    axes[1].plot(steps, val_acc, label="val acc", color="#8e44ad", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        f"Modular Addition Grokking (M={cfg.modulus}, train_fraction={cfg.train_fraction}, wd={cfg.weight_decay})",
        fontsize=11,
    )
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    x_all, y_all, _ = build_modular_addition_data(cfg.modulus)
    x_train, y_train, x_val, y_val = split_train_val(x_all, y_all, cfg.train_fraction, rng)

    model = MLP(input_dim=2 * cfg.modulus, hidden_sizes=cfg.hidden_sizes, output_dim=cfg.modulus, rng=rng)

    metrics: list[dict[str, float]] = []
    memorization_step = None
    generalization_step = None

    for step in range(1, cfg.steps + 1):
        if cfg.batch_size >= x_train.shape[0]:
            x_batch = x_train
            y_batch = y_train
        else:
            batch_idx = rng.integers(0, x_train.shape[0], size=cfg.batch_size)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

        _ = model.step(x_batch, y_batch, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        if step % cfg.eval_every == 0 or step == 1:
            _, _, _, train_probs = model.forward(x_train)
            _, _, _, val_probs = model.forward(x_val)

            eps = 1e-9
            train_loss = float(-np.mean(np.sum(y_train * np.log(train_probs + eps), axis=1)))
            val_loss = float(-np.mean(np.sum(y_val * np.log(val_probs + eps), axis=1)))
            train_acc = accuracy(train_probs, y_train)
            val_acc = accuracy(val_probs, y_val)

            if memorization_step is None and train_acc >= 0.99:
                memorization_step = step
            if generalization_step is None and val_acc >= 0.99:
                generalization_step = step

            metrics.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metrics_csv(out_dir / "metrics.csv", metrics)
    plot_metrics(out_dir / "curves.png", metrics, cfg)

    summary = {
        "config": asdict(cfg),
        "num_total_examples": int(x_all.shape[0]),
        "num_train_examples": int(x_train.shape[0]),
        "num_val_examples": int(x_val.shape[0]),
        "final": metrics[-1],
        "memorization_step": memorization_step,
        "generalization_step": generalization_step,
        "grokking_gap": None if (memorization_step is None or generalization_step is None) else int(generalization_step - memorization_step),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
