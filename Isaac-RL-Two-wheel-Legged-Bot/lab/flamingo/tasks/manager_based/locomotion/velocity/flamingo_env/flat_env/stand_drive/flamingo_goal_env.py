# flamingo_goal_env.py - NEW FILE
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject

class FlamingoGoalEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Target management
        self.target_radius = 0.5
        self.target_spawn_range = 5.0
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize random targets
        self.respawn_all_targets()
    
    def respawn_all_targets(self):
        """Spawn new random targets for all environments."""
        random_x = torch.uniform(-self.target_spawn_range, self.target_spawn_range, (self.num_envs,), device=self.device)
        random_y = torch.uniform(-self.target_spawn_range, self.target_spawn_range, (self.num_envs,), device=self.device)
        
        self.target_positions[:, 0] = random_x
        self.target_positions[:, 1] = random_y
        self.target_positions[:, 2] = 0.1
        
        # Update visual markers
        if "target" in self.scene:
            target_object: RigidObject = self.scene["target"]
            target_object.write_root_pose_to_sim(
                position=self.target_positions,
                orientation=None
            )
    
    def respawn_targets(self, mask: torch.Tensor):
        """Respawn targets for successful environments."""
        if not torch.any(mask):
            return
            
        num_respawn = torch.sum(mask)
        new_x = torch.uniform(-self.target_spawn_range, self.target_spawn_range, (num_respawn,), device=self.device)
        new_y = torch.uniform(-self.target_spawn_range, self.target_spawn_range, (num_respawn,), device=self.device)
        
        self.target_positions[mask, 0] = new_x
        self.target_positions[mask, 1] = new_y
        
        # Update visual markers
        if "target" in self.scene:
            target_object: RigidObject = self.scene["target"]
            target_object.write_root_pose_to_sim(
                position=self.target_positions,
                orientation=None
            )