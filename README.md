# Grokking Experiment Reproduction Framework

A config-driven framework for reproducing the key experiments from **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets** (Power et al., 2021):

- Modular arithmetic classification (`a op b mod p`)
- Permutation group composition (`S_n`)

The framework is lightweight (NumPy MLP + AdamW/SGD), deterministic by seed, and writes reproducible artifacts for each run.

## What You Get

- Reusable experiment package in `src/grokking`
- Built-in presets for baseline vs weight-decay (grokking-friendly) setups
- Single-run CLI (`run_experiment.py`)
- Multi-seed sweep CLI (`run_suite.py`)
- JSON configs under `configs/`
- Artifacts per run:
  - `config.json`
  - `metrics.csv`
  - `summary.json`
  - `curves.png` (if `matplotlib` is available)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

List presets:

```bash
python3 run_experiment.py --list-presets
```

Run modular baseline:

```bash
python3 run_experiment.py \
  --preset mod_add_baseline \
  --out artifacts/mod_add_baseline
```

Run modular grokking-style setup (weight decay enabled):

```bash
python3 run_experiment.py \
  --preset mod_add_grokking \
  --out artifacts/mod_add_grokking
```

Run with config file:

```bash
python3 run_experiment.py \
  --config configs/perm_s5_grokking.json \
  --out artifacts/perm_s5_grokking
```

Override settings from CLI:

```bash
python3 run_experiment.py \
  --preset mod_add_grokking \
  --override training.steps=20000 \
  --override training.seed=3 \
  --override optimizer.weight_decay=0.02 \
  --no-curves \
  --out artifacts/custom_run
```

## Run Paper-Style Sweeps

Run all built-in presets across multiple seeds:

```bash
python3 run_suite.py \
  --presets mod_add_baseline,mod_add_grokking,perm_s5_baseline,perm_s5_grokking \
  --seeds 0,1,2 \
  --no-curves \
  --out artifacts/sweeps/paper_suite
```

This writes:

- `artifacts/sweeps/paper_suite/sweep_summary.csv`
- `artifacts/sweeps/paper_suite/sweep_summary.json`
- Per-run artifact folders at `.../<preset>/seed_<k>/`

## Presets

- `mod_add_baseline`
- `mod_add_grokking`
- `perm_s5_baseline`
- `perm_s5_grokking`

## Project Layout

- `run_experiment.py`: single run entrypoint
- `run_suite.py`: multi-seed sweep entrypoint
- `configs/*.json`: reproducible experiment definitions
- `src/grokking/config.py`: schema + presets + overrides
- `src/grokking/tasks.py`: modular/permutation dataset builders
- `src/grokking/model.py`: NumPy MLP classifier
- `src/grokking/optimizer.py`: AdamW and SGD
- `src/grokking/experiment.py`: train/eval loop + artifact writing
- `src/grokking/sweeps.py`: sweep orchestration
- `tests/`: task invariants + smoke tests

## Testing

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Reproducibility Notes

- Runs are deterministic by `training.seed`.
- Train/val split is generated once per run from the full Cartesian task table.
- Grokking timing is sensitive to optimization hyperparameters and seed. For delayed generalization to clearly appear, increase `training.steps` and compare baseline vs weight-decay presets across multiple seeds.
