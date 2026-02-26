from __future__ import annotations

from typing import Any

import numpy as np


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        seed: int = 0,
    ) -> None:
        self.activation = activation.lower()
        if self.activation not in {"relu", "tanh"}:
            raise ValueError("activation must be one of: relu, tanh")

        dims = [input_dim, *hidden_dims, output_dim]
        rng = np.random.default_rng(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.biases.append(np.zeros((fan_out,), dtype=np.float32))

    def named_parameters(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out[f"W{i}"] = w
            out[f"b{i}"] = b
        return out

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, x)
        return np.tanh(x)

    def _act_grad(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (x > 0.0).astype(np.float32)
        t = np.tanh(x)
        return (1.0 - t * t).astype(np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        acts: list[np.ndarray] = [x]
        preacts: list[np.ndarray] = []

        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            preacts.append(z)
            if i == len(self.weights) - 1:
                h = z
            else:
                h = self._act(z)
            acts.append(h)

        return h, {"acts": acts, "preacts": preacts}

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        logits, cache = self.forward(x)
        logits = logits.astype(np.float64)

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        n = y.shape[0]
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        grad = probs
        grad[np.arange(n), y] -= 1.0
        grad /= n
        grad = grad.astype(np.float32)

        grads: dict[str, np.ndarray] = {}

        for i in reversed(range(len(self.weights))):
            a_prev = cache["acts"][i]
            w = self.weights[i]

            grad_w = a_prev.T @ grad
            grad_b = grad.sum(axis=0)

            grads[f"W{i}"] = grad_w.astype(np.float32)
            grads[f"b{i}"] = grad_b.astype(np.float32)

            if i > 0:
                grad = (grad @ w.T) * self._act_grad(cache["preacts"][i - 1])

        return float(loss), grads, logits.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)
