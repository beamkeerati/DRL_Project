#!/bin/bash

# Set the current directory as the base path
BASE_DIR="$(pwd)"

# Set up the Python path correctly
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Run with correct arguments and append parameters with +
python "${BASE_DIR}/scripts/co_rl/train.py" \
  --task Isaac-Velocity-Flat-Flamingo-v1-ppo \
  --algo ppo \
  --num_envs 4096 \
#   --headless \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  +learning_rate=0.0003 \
  +num_learning_epochs=10 \
  +num_mini_batches=8 \
  +entropy_coef=0.01