# scripts/co_rl/core/wrapper/residual_wrapper.py

import torch
from scripts.co_rl.core.modules.pd_controller import PDController

class ResidualRLWrapper:
    def __init__(
        self,
        wrapper_env,
        base_env,
        pd_kp=None,
        pd_kd=None,
        residual_scale=0.3,
        pd_ratio=0.7,
        asset_name="robot",
        # Add curriculum parameters
        curriculum_enabled=False,
        initial_pd_ratio=0.9,
        final_pd_ratio=0.0,  # 0.0 for pure RL at the end
        curriculum_steps=1000,
    ):
        self.wrapper_env = wrapper_env
        self.base_env = base_env
        self.residual_scale = residual_scale
        self.pd_ratio = pd_ratio
        self.device = base_env.device
        self.asset_name = asset_name
        
        # Curriculum learning parameters
        self.curriculum_enabled = curriculum_enabled
        self.initial_pd_ratio = initial_pd_ratio
        self.final_pd_ratio = final_pd_ratio
        self.curriculum_steps = curriculum_steps
        self.current_iteration = 0
        
        # If curriculum is enabled, start with initial pd_ratio
        if self.curriculum_enabled:
            self.pd_ratio = self.initial_pd_ratio

        # Rest of your initialization code...
        self.action_dim = wrapper_env.action_space.shape[0]
        self.pd_controlled_joint_indices = list(range(self.action_dim))
        self.pd_controlled_joint_indices = [i for i in self.pd_controlled_joint_indices if i < 8]
        self.residual_joint_indices = list(range(8))

        self.pd_controller = PDController(
            num_actions=len(self.pd_controlled_joint_indices),
            kp=pd_kp,
            kd=pd_kd,
            device=self.device,
        )
        
        # Debug information
        print(f"[ResidualRLWrapper] Initialized with:")
        print(f"  - Action dimension: {self.action_dim}")
        print(f"  - PD controlled joints: {len(self.pd_controlled_joint_indices)} joints")
        print(f"  - Residual joints: {len(self.residual_joint_indices)} joints")
        print(f"  - PD ratio: {self.pd_ratio}, Residual scale: {self.residual_scale}")
        print(f"  - Asset name: {self.asset_name}")
        if self.curriculum_enabled:
            print(f"  - Curriculum enabled: {self.initial_pd_ratio} -> {self.final_pd_ratio} over {self.curriculum_steps} iterations")

    def update_curriculum(self, current_iteration):
        """Update PD ratio based on current training iteration using linear decay."""
        if not self.curriculum_enabled:
            return
        
        self.current_iteration = current_iteration
        
        # Linear interpolation from initial to final pd_ratio
        if current_iteration >= self.curriculum_steps:
            self.pd_ratio = self.final_pd_ratio
        else:
            progress = current_iteration / self.curriculum_steps
            self.pd_ratio = self.initial_pd_ratio + (self.final_pd_ratio - self.initial_pd_ratio) * progress
        
        # Optional: print progress every N iterations
        if current_iteration % 100 == 0:
            print(f"[ResidualRLWrapper] Iteration {current_iteration}: PD ratio = {self.pd_ratio:.3f}")

    def step(self, rl_actions):
        # Your existing step implementation remains the same
        if rl_actions.shape[1] != len(self.residual_joint_indices):
            raise ValueError(f"RL actions dimension mismatch: expected {len(self.residual_joint_indices)}, got {rl_actions.shape[1]}")
        
        # Get joint states
        joint_pos = None
        joint_vel = None
        
        if hasattr(self.base_env.scene, self.asset_name):
            asset = getattr(self.base_env.scene, self.asset_name)
            if hasattr(asset, "data"):
                joint_pos = asset.data.joint_pos
                joint_vel = asset.data.joint_vel
        
        if joint_pos is None and hasattr(self.base_env.scene, "__getitem__"):
            try:
                asset = self.base_env.scene[self.asset_name]
                if hasattr(asset, "data"):
                    joint_pos = asset.data.joint_pos
                    joint_vel = asset.data.joint_vel
            except:
                pass
        
        if joint_pos is None or joint_vel is None:
            raise ValueError(
                f"Cannot access joint states from asset '{self.asset_name}'. "
                "Please check your asset name in the environment scene."
            )

        # Extract PD controlled joints
        pd_joint_pos = joint_pos[:, self.pd_controlled_joint_indices]
        pd_joint_vel = joint_vel[:, self.pd_controlled_joint_indices]

        pd_actions = self.pd_controller.compute_pd_action(pd_joint_pos, pd_joint_vel)

        # Apply residual actions
        full_residual_actions = torch.zeros_like(pd_actions)
        full_residual_actions[:, self.residual_joint_indices] = rl_actions * self.residual_scale

        # Combine PD and RL actions using the (potentially updated) pd_ratio
        combined = self.pd_ratio * pd_actions + (1 - self.pd_ratio) * full_residual_actions
        combined = torch.clamp(combined, -1.0, 1.0)

        # Call wrapper's step
        self.wrapper_env.residual_mode = False
        policy_obs, reward, done, extras = self.wrapper_env.step(combined)
        self.wrapper_env.residual_mode = True
        
        return policy_obs, reward, done, extras

    def reset(self, *args, **kwargs):
        """Reset the environment and return observation tensor."""
        # Call the wrapper's reset to ensure proper observation stacking
        return self.wrapper_env.reset(*args, **kwargs)

    def get_current_obs(self):
        """
        Fetch current policy-stacked observations from the wrapper.

        Returns:
            obs (torch.Tensor): The policy observations with proper dimensions
        """
        obs, _ = self.wrapper_env.get_observations()
        return obs

    def __getattr__(self, name):
        """Forward any other attribute to the base environment."""
        return getattr(self.base_env, name)

def move_obs_to_device(obs, device):
    if isinstance(obs, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
    elif isinstance(obs, torch.Tensor):
        return obs.to(device)
    else:
        return obs