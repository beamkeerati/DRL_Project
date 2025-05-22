from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg

@configclass
class FlamingoFlatSACRunnerCfg(CoRlPolicyRunnerCfg):
    """Runner config for SAC on FlamingoFlatEnv."""
    algorithm: dict = {
        "class_name": "SAC",
        "module": "scripts.co_rl.core.algorithms.sac",
    }

    # ─── HARD-CODE THE ACTUAL OBS SHAPE ─────────────────────────────
    # Based on your env’s observation_space, this must be 159.
    state_shape: tuple = (159,)
    # ────────────────────────────────────────────────────────────────

    actor_lr: float  = 3e-4
    critic_lr: float = 3e-4
    tau: float       = 0.005
    batch_size: int  = 256
    gamma: float     = 0.99
