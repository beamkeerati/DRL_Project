#!/bin/bash

# Set up the Python path 
export PYTHONPATH=$PWD:$PWD/scripts:$PWD/scripts/co_rl

# Run with correct arguments
python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo --algo ppo --num_envs 4 --headless --num_policy_stacks 2 --num_critic_stacks 2
