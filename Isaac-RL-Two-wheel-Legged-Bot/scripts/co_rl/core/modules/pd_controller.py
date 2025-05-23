# scripts/co_rl/core/modules/pd_controller.py
import torch
import torch.nn as nn

class PDController(nn.Module):
    """PD controller for robot joints with residual RL support"""
    
    def __init__(self, num_actions, kp=None, kd=None, device='cuda'):
        super().__init__()
        self.num_actions = num_actions
        self.device = device
        
        # Default PD gains (uniform values)
        if kp is None:
            self.kp = torch.full((num_actions,), 100.0, device=device)
        else:
            self.kp = torch.tensor(kp, device=device)
            
        if kd is None:
            self.kd = torch.full((num_actions,), 10.0, device=device)
        else:
            self.kd = torch.tensor(kd, device=device)
            
        # Default joint positions (all zeros)
        self.default_pos = torch.zeros(num_actions, device=device)
        
    def compute_pd_action(self, joint_pos, joint_vel, target_pos=None):
        """
        Compute PD control action
        Args:
            joint_pos: Current joint positions (batch_size, num_actions)
            joint_vel: Current joint velocities (batch_size, num_actions)
            target_pos: Target positions (batch_size, num_actions) or None for default
        Returns:
            pd_actions: PD control outputs (batch_size, num_actions)
        """
        if target_pos is None:
            target_pos = self.default_pos.unsqueeze(0).expand(joint_pos.shape[0], -1)
            
        pos_error = target_pos - joint_pos
        pd_actions = self.kp * pos_error - self.kd * joint_vel
        
        # Normalize using uniform max torque (adjust if needed)
        max_torques = torch.full((self.num_actions,), 100.0, device=self.device)
        pd_actions = torch.clamp(pd_actions / max_torques, -1.0, 1.0)
        
        return pd_actions