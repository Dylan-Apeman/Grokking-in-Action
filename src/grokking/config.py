from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
from typing import Any


@dataclass
class TaskConfig:
    kind: str = "modular"  # modular | permutation
    operation: str = "add"  # add | sub | mul
    modulus: int = 97
    permutation_size: int = 5


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [2048, 2048, 2048])
    activation: str = "relu"  # relu | tanh


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    steps: int = 50_000
    batch_size: int = 512
    train_fraction: float = 0.3
    seed: int = 0
    eval_every: int = 100
    metric_threshold: float = 0.99


@dataclass
class LoggingConfig:
    verbose: bool = True
    save_curves: bool = True


@dataclass
class ExperimentConfig:
    name: str
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(data: dict[str, Any]) -> ExperimentConfig:
    task = TaskConfig(**data.get("task", {}))
    model = ModelConfig(**data.get("model", {}))
    optimizer = OptimizerConfig(**data.get("optimizer", {}))
    training = TrainingConfig(**data.get("training", {}))
    logging = LoggingConfig(**data.get("logging", {}))
    name = data.get("name", "unnamed")
    return ExperimentConfig(
        name=name,
        task=task,
        model=model,
        optimizer=optimizer,
        training=training,
        logging=logging,
    )


def load_config_file(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _build_config(data)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    data = config.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        path, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        keys = path.split(".")
        target = data
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Override path not found: {path}")
            target = target[key]
        leaf_key = keys[-1]
        if leaf_key not in target:
            raise KeyError(f"Override path not found: {path}")
        target[leaf_key] = value
    return _build_config(data)


def _preset_catalog() -> dict[str, ExperimentConfig]:
    base_training = TrainingConfig(steps=50_000, batch_size=512, train_fraction=0.3, eval_every=100)

    mod_base = ExperimentConfig(
        name="mod_add_baseline",
        task=TaskConfig(kind="modular", operation="add", modulus=97),
        model=ModelConfig(hidden_dims=[4096, 4096, 4096], activation="relu"),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0),
        training=copy.deepcopy(base_training),
    )

    mod_grok = ExperimentConfig(
        name="mod_add_grokking",
        task=TaskConfig(kind="modular", operation="add", modulus=97),
        model=ModelConfig(hidden_dims=[4096, 4096, 4096], activation="relu"),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-2),
        training=copy.deepcopy(base_training),
    )

    perm_base = ExperimentConfig(
        name="perm_s5_baseline",
        task=TaskConfig(kind="permutation", permutation_size=5),
        model=ModelConfig(hidden_dims=[4096, 4096, 4096], activation="relu"),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0),
        training=TrainingConfig(steps=80_000, batch_size=512, train_fraction=0.5, eval_every=200),
    )

    perm_grok = ExperimentConfig(
        name="perm_s5_grokking",
        task=TaskConfig(kind="permutation", permutation_size=5),
        model=ModelConfig(hidden_dims=[4096, 4096, 4096], activation="relu"),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=3e-3),
        training=TrainingConfig(steps=80_000, batch_size=512, train_fraction=0.5, eval_every=200),
    )

    return {
        mod_base.name: mod_base,
        mod_grok.name: mod_grok,
        perm_base.name: perm_base,
        perm_grok.name: perm_grok,
    }


def get_preset_config(name: str) -> ExperimentConfig:
    catalog = _preset_catalog()
    if name not in catalog:
        available = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return copy.deepcopy(catalog[name])


def list_presets() -> list[str]:
    return sorted(_preset_catalog().keys())
