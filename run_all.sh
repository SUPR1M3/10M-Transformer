#!/bin/bash
# Queue of sweeps, in priority order. Each step skips runs already in results/runs.csv.
LOG=/content/drive/MyDrive/jax-addition/sweep.log
run() { echo "=== $(date) START $*"; env "$@" python -u run_sweep.py; echo "=== $(date) END $*"; }
{
  run ARCH=dense SEED=0                 # 1. finish dense: 4 missing 1e15 runs (~17 min)
  run ARCH=moe   SEED=0                 # 2. MoE sweep (~3-4 h; MoE is slower per token)
  run ARCH=dense SEED=0 MAX_TOKENS=9e8  # 3. dense left-side fill-ins: 4 runs (~1.6 h)
  run ARCH=moe   SEED=0 MAX_TOKENS=9e8  # 4. MoE left-side fill-ins (~1.6-2 h)
} >> "$LOG" 2>&1