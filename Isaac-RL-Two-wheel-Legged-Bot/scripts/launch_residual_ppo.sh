#!/usr/bin/env bash
# =============================================================================
# Launch a residual-augmented PPO run for Flamingo Flat Stand-Drive env
# =============================================================================

# 1) Always include your project root in PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 2) Dispatch to the CO-RL trainer from your project root
python scripts/co_rl/train.py \
    --task Isaac-Velocity-Flat-Flamingo-v1-ppo \
    --algo ppo \
    --num_envs 64 \
    --num_policy_stacks 2 \
    --num_critic_stacks 2 \
    --max_iterations 5000 \
    --residual \
    --residual_scale 0.3 \
    --pd_ratio 0.5 \
    --run_name residual_ppo\
    +learning_rate=0.0003 \
    +num_learning_epochs=10 \
    +num_mini_batches=8 \
    +entropy_coef=0.01