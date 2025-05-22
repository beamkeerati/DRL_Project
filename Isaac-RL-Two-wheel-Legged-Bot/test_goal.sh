#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# Run your Flamingo goal‐seeking RL entirely inside Isaac Sim
# ───────────────────────────────────────────────────────────

# 1) Ensure your code repo is on PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 2) Path to your Isaac Sim launcher
ISAAC_SIM=~/Omniverse/IsaacSim/isaac-sim.sh

# 3) Preload the registration module inside Isaac Sim
$ISAAC_SIM <<'PYCODE'
import lab.flamingo.tasks.manager_based.locomotion.velocity.\
flamingo_env.flat_env.stand_drive  # noqa: F401
PYCODE

# 4) Run the trainer inside Isaac Sim
$ISAAC_SIM scripts/co_rl/train.py \
  --task Isaac-Goal-Flat-Flamingo-v1-ppo \
  --algo ppo \
  --num_envs 64 \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --resume True \
  --load_run ver1 \
  --max_iterations 20000 \
  +learning_rate=0.0003 \
  +num_learning_epochs=10 \
  +num_mini_batches=8 \
  +entropy_coef=0.01
