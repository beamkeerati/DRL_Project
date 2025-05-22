#!/usr/bin/env bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

TASK="Isaac-Velocity-Flat-Flamingo-v3-sac"   # or whatever your exact SAC ID is
NUM_ENVS=64
MAX_ITERS=5000

python scripts/co_rl/train.py \
  --task ${TASK} \
  --algo sac \
  --num_envs ${NUM_ENVS} \
  --max_iterations ${MAX_ITERS} \
  +actor_lr=0.0003 \
  +critic_lr=0.0003 \
  +tau=0.005 \
  +batch_size=256 \
  +gamma=0.99
