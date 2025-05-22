#!/usr/bin/env bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ───────────────────────────────────────────────────────────────
# 1) Point at the run folder under logs/co_rl/.../ppo
#    (use the timestamped directory name exactly)
RUN="ppo"

# 2) Resume hyperparameters
NUM_ENVS=64
POLICY_STACKS=2
CRITIC_STACKS=2
MAX_ITERS=25000

# 3) Invoke the trainer
python scripts/co_rl/train.py \
  --task Isaac-Velocity-Flat-Flamingo-v1-ppo \
  --algo ppo \
  --num_envs ${NUM_ENVS} \
  --num_policy_stacks ${POLICY_STACKS} \
  --num_critic_stacks ${CRITIC_STACKS} \
  --resume True \
  --load_run ${RUN} \
  --max_iterations ${MAX_ITERS} \
  +learning_rate=0.0003 \
  +num_learning_epochs=10 \
  +num_mini_batches=8 \
  +entropy_coef=0.01
