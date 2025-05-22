import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

def distance_to_target_exp(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Exponential reward for being close to target."""
    # Get robot position
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]  # x, y only
    
    # Simple fixed target for now (you can make this dynamic later)
    target_pos = torch.tensor([3.0, 0.0], device=env.device).expand(env.num_envs, -1)
    
    # Distance to target
    distance = torch.norm(robot_pos - target_pos, dim=1)
    
    # Exponential reward: closer = much higher reward
    reward = torch.exp(-distance * 1.0)
    return reward

def reached_target_bonus(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Large bonus for reaching target."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]
    
    target_pos = torch.tensor([3.0, 0.0], device=env.device).expand(env.num_envs, -1)
    distance = torch.norm(robot_pos - target_pos, dim=1)
    
    # Check which robots reached target
    reached = distance < 0.5
    
    return reached.float() * 10.0  # Success bonus