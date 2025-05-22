#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/co_rl/train.py \
  --task Isaac-Velocity-Flat-Flamingo-v1-ppo \
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