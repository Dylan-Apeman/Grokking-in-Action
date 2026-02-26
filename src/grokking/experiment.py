from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .model import MLPClassifier
from .optimizer import build_optimizer
from .tasks import build_dataset, encode_pairs_one_hot, split_dataset


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = labels.shape[0]
    return float(-np.log(probs[np.arange(n), labels] + 1e-12).mean())


def _accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    pred = np.argmax(logits, axis=1)
    return float((pred == labels).mean())


def _evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray, batch_size: int = 4096) -> tuple[float, float]:
    if y.size == 0:
        return float("nan"), float("nan")

    losses = []
    correct = 0
    total = y.shape[0]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits, _ = model.forward(x[start:end])
        batch_y = y[start:end]
        losses.append(_cross_entropy_from_logits(logits.astype(np.float64), batch_y))
        correct += int((np.argmax(logits, axis=1) == batch_y).sum())

    return float(np.mean(losses)), float(correct / total)


def _first_step_above(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in rows:
        value = row.get(key)
        if isinstance(value, float) and not np.isnan(value) and value >= threshold:
            return int(row["step"])
    return None


def _save_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_curves(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    steps = [r["step"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(steps, train_loss, label="train")
    ax[0].plot(steps, val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Cross-entropy")
    ax[0].legend()

    ax[1].plot(steps, train_acc, label="train")
    ax[1].plot(steps, val_acc, label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def run_experiment(config: ExperimentConfig, out_dir: str | Path) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    task = build_dataset(config.task)
    split = split_dataset(
        task.pairs,
        task.labels,
        train_fraction=config.training.train_fraction,
        seed=config.training.seed,
    )

    x_train = encode_pairs_one_hot(split.train_pairs, task.vocab_size)
    x_val = encode_pairs_one_hot(split.val_pairs, task.vocab_size)

    model = MLPClassifier(
        input_dim=x_train.shape[1],
        hidden_dims=config.model.hidden_dims,
        output_dim=task.num_classes,
        activation=config.model.activation,
        seed=config.training.seed,
    )
    optimizer = build_optimizer(config.optimizer)

    rng = np.random.default_rng(config.training.seed + 17)

    rows: list[dict[str, Any]] = []
    n_train = x_train.shape[0]

    for step in range(1, config.training.steps + 1):
        batch_idx = rng.integers(0, n_train, size=config.training.batch_size)
        xb = x_train[batch_idx]
        yb = split.train_labels[batch_idx]

        _, grads, _ = model.loss_and_grads(xb, yb)
        optimizer.step(model.named_parameters(), grads)

        if step == 1 or step % config.training.eval_every == 0 or step == config.training.steps:
            train_loss, train_acc = _evaluate(model, x_train, split.train_labels)
            val_loss, val_acc = _evaluate(model, x_val, split.val_labels)

            row = {
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            rows.append(row)

            if config.logging.verbose:
                print(
                    f"step={step:>7d} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

    memorization_step = _first_step_above(rows, "train_acc", config.training.metric_threshold)
    generalization_step = _first_step_above(rows, "val_acc", config.training.metric_threshold)

    final = rows[-1]
    summary = {
        "name": config.name,
        "task": task.metadata,
        "train_size": int(split.train_labels.shape[0]),
        "val_size": int(split.val_labels.shape[0]),
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_val_loss": final["val_loss"],
        "final_val_acc": final["val_acc"],
        "memorization_step": memorization_step,
        "generalization_step": generalization_step,
        "threshold": config.training.metric_threshold,
        "steps": config.training.steps,
        "runtime_sec": round(time.time() - t0, 3),
    }

    _save_metrics_csv(rows, out_path / "metrics.csv")
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    curve_saved = False
    if config.logging.save_curves:
        curve_saved = _save_curves(rows, out_path / "curves.png")
    summary["curves_saved"] = curve_saved

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
