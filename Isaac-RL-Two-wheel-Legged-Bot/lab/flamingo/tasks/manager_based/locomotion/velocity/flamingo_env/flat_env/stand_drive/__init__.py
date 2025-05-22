import gymnasium as gym
from .flat_env_stand_drive_cfg import FlamingoFlatEnvCfg
from .agents.co_rl_cfg import FlamingoFlatSACRunnerCfg

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v3-sac",   # e.g. "Isaac-Velocity-Flat-Flamingo-v3-sac"
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": f"{__name__}.agents.co_rl_cfg:FlamingoFlatSACRunnerCfg",
    },
)
