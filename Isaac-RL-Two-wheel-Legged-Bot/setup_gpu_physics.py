import os
import sys
import torch

# Set environment variables
os.environ["ENABLE_GPU_PHYSICS"] = "1"
os.environ["ENABLE_FABRIC_SHARED_MEMORY"] = "1"
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.getcwd()}/scripts:{os.getcwd()}/scripts/co_rl"

# Print current configuration
print(f"Current directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Now execute the training script
cmd = "python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo --algo ppo --num_envs 4 --num_policy_stacks 2 --num_critic_stacks 2 --sim_device cuda:0 --rl_device cuda:0 --pipeline gpu"
print(f"Executing: {cmd}")
os.system(cmd)
