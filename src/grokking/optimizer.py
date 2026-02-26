from __future__ import annotations

import abc
import numpy as np

from .config import OptimizerConfig


def _should_decay(name: str) -> bool:
    return name.startswith("W")


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float, weight_decay: float = 0.0, momentum: float = 0.0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for name, p in params.items():
            g = grads[name]

            if self.weight_decay > 0.0 and _should_decay(name):
                p *= (1.0 - self.lr * self.weight_decay)

            if self.momentum > 0.0:
                v = self.velocity.get(name)
                if v is None:
                    v = np.zeros_like(p)
                v = self.momentum * v + g
                self.velocity[name] = v
                p -= self.lr * v
            else:
                p -= self.lr * g


class AdamW(Optimizer):
    def __init__(
        self,
        lr: float,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1

        for name, p in params.items():
            g = grads[name]

            if self.weight_decay > 0.0 and _should_decay(name):
                p -= self.lr * self.weight_decay * p

            m = self.m.get(name)
            v = self.v.get(name)
            if m is None:
                m = np.zeros_like(p)
            if v is None:
                v = np.zeros_like(p)

            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            self.m[name] = m
            self.v[name] = v

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def build_optimizer(config: OptimizerConfig) -> Optimizer:
    name = config.name.lower()
    if name == "sgd":
        return SGD(lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    if name == "adamw":
        return AdamW(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")
