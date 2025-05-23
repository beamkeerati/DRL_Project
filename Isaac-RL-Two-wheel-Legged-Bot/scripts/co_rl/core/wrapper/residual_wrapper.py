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
        asset_name="robot",  # Add asset name parameter
    ):
        self.wrapper_env = wrapper_env
        self.base_env = base_env
        self.residual_scale = residual_scale
        self.pd_ratio = pd_ratio
        self.device = base_env.device
        self.asset_name = asset_name  # Store asset name

        # Total action dimension from environment action space
        self.action_dim = wrapper_env.action_space.shape[0]

        # === IMPORTANT ===
        # Define indices of joints that PD controller acts on
        # (should match the PD controller expected joints, e.g., 44 joints)
        self.pd_controlled_joint_indices = list(range(self.action_dim))  # Change if needed
        self.pd_controlled_joint_indices = [i for i in self.pd_controlled_joint_indices if i < 8]
        

        # Define indices of joints the residual policy outputs residuals for
        # (for example, residuals for 8 specific joints)
        self.residual_joint_indices = list(range(8))  # Change to your residual joints indices

        # Initialize PD controller with number of controlled joints
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
        
        # Debug: Print available assets in the scene
        if hasattr(self.base_env, "scene"):
            print(f"[ResidualRLWrapper] Available scene attributes:")
            for attr in dir(self.base_env.scene):
                if not attr.startswith("_"):
                    obj = getattr(self.base_env.scene, attr)
                    if hasattr(obj, "data"):
                        print(f"  - {attr}: {type(obj)}")

    def step(self, rl_actions):
        # Debug: Check RL actions dimension
        if rl_actions.shape[1] != len(self.residual_joint_indices):
            raise ValueError(f"RL actions dimension mismatch: expected {len(self.residual_joint_indices)}, got {rl_actions.shape[1]}")
        
        # Get raw joint states directly from the environment for PD control
        joint_pos = None
        joint_vel = None
        
        # First try to use the configured asset name
        if hasattr(self.base_env.scene, self.asset_name):
            asset = getattr(self.base_env.scene, self.asset_name)
            if hasattr(asset, "data"):
                joint_pos = asset.data.joint_pos
                joint_vel = asset.data.joint_vel
        
        # If not found, try accessing through scene dictionary
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

        # Validate indices before indexing
        max_idx_pos = joint_pos.shape[1] - 1
        max_idx_vel = joint_vel.shape[1] - 1

        if max(self.pd_controlled_joint_indices) > max_idx_pos or max(self.pd_controlled_joint_indices) > max_idx_vel:
            raise IndexError(f"pd_controlled_joint_indices out of bounds: max index is "
                            f"{max(self.pd_controlled_joint_indices)}, but max allowed is {min(max_idx_pos, max_idx_vel)}")

        # Extract the joints that PD controller acts on
        pd_joint_pos = joint_pos[:, self.pd_controlled_joint_indices]
        pd_joint_vel = joint_vel[:, self.pd_controlled_joint_indices]

        pd_actions = self.pd_controller.compute_pd_action(pd_joint_pos, pd_joint_vel)

        # Validate residual indices similarly
        max_idx_pd = len(self.pd_controlled_joint_indices) - 1
        if max(self.residual_joint_indices) > max_idx_pd:
            raise IndexError(f"residual_joint_indices out of bounds: max index is "
                            f"{max(self.residual_joint_indices)}, but max allowed is {max_idx_pd}")

        full_residual_actions = torch.zeros_like(pd_actions)
        full_residual_actions[:, self.residual_joint_indices] = rl_actions * self.residual_scale

        combined = self.pd_ratio * pd_actions + (1 - self.pd_ratio) * full_residual_actions
        combined = torch.clamp(combined, -1.0, 1.0)

        # IMPORTANT: Call the wrapper's original step function, not the base env's step
        # This ensures proper observation stacking and processing
        # We temporarily disable residual mode to avoid infinite recursion
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