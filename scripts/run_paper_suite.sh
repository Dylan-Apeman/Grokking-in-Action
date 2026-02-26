#!/usr/bin/env bash
set -euo pipefail

python3 run_suite.py \
  --presets mod_add_baseline,mod_add_grokking,perm_s5_baseline,perm_s5_grokking \
  --seeds 0,1,2 \
  --out artifacts/paper_suite
