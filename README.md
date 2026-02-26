# Grokking in Modular Arithmetic

A portfolio-ready reproduction of the **grokking** phenomenon from the 2021 paper *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* by Power et al. by OpenAI and Google DeepMind researchers. (https://arxiv.org/abs/2201.02177).

The core logical components of the model trained herein are: 

- one-hot encoded modular arithmetic inputs
- MLP trained with manual NumPy backprop
- train/validation dynamics tracked over long optimization

## What this shows

`grokking` is delayed generalization: the model can memorize training examples early, while validation performance stays near chance for a long period, then sharply improves later.

This repo provides two presets:

- `baseline`: no weight decay (tends to memorize and generalize poorly)
- `grokking`: weight decay enabled (encourages delayed generalization)

## Project structure

- `run_experiment.py`: CLI runner with presets
- `src/grokking/experiment.py`: dataset, model, training loop, plotting, metrics
- `test.py`: short compatibility demo entrypoint
- `tests/test_experiment.py`: unit tests for task/data invariants
- `artifacts/`: generated metrics, summary JSON, and plots

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run baseline:

```bash
python3 run_experiment.py --preset baseline --steps 40000 --out artifacts/baseline
```

Run grokking setup:

```bash
python3 run_experiment.py --preset grokking --steps 40000 --out artifacts/grokking
```

Run a short demo (legacy `test.py`):

```bash
python3 test.py
```

Run tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Artifacts produced

Each run writes:

- `metrics.csv`: step-wise train/val loss and accuracy
- `summary.json`: final metrics + detected memorization/generalization steps
- `curves.png`: loss and accuracy curves for portfolio screenshots
  - If `matplotlib` is unavailable, training still runs and exports `metrics.csv` + `summary.json`.

## Results (40k steps)

Side-by-side comparison from full runs:

![Baseline vs Grokking (40k steps)](artifacts/results/baseline_vs_grokking_40k.png)

Run summary (`M=97`, `train_fraction=0.3`, `seed=0`):

| Setup | Weight Decay | Train Acc | Val Acc | Memorization Step | Generalization Step |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0 | 1.000 | 0.000 | 4600 | not reached |
| Grokking preset | 0.001 | 1.000 | 0.000 | 7800 | not reached |

In this seed/hyperparameter setting, both models fully memorize training data by 40k steps, but delayed generalization is not yet observed.

## Notes on reproducibility

- Default task: modular addition with modulus `97`.
- Data split: random subset (`train_fraction=0.3`) of all pairs.
- Minibatch SGD is used (`batch_size=512` by default).
- Grokking timing is sensitive to seed and hyperparameters; if needed, increase `--steps`.
