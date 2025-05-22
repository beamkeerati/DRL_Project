# flamingo_goal_env.py (Pure Isaac Gym)

from isaacgym import gymapi, gymtorch
import torch
from .vec_env import VecEnv  # your vectorized‐env base

class FlamingoGoalEnv(VecEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 1) Sphere asset and actor handles:
        opts = gymapi.AssetOptions()
        self.sphere_asset = self.gym.create_sphere(self.sim, radius=0.2, options=opts)
        self.sphere_handles = []
        for env in self.envs:
            h = self.gym.create_actor(env, self.sphere_asset,
                                      gymapi.Transform(), "target", 0)
            self.sphere_handles.append(h)
        # 2) Target positions tensor:
        self.target_positions = torch.zeros(self.num_envs, 2, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)  # reset robots, etc.
        # 3) Randomize targets in [-3,3]^2:
        n = len(env_ids)
        xs = torch.rand(n, device=self.device) * 6.0 - 3.0
        ys = torch.rand(n, device=self.device) * 6.0 - 3.0
        self.target_positions[env_ids, 0] = xs
        self.target_positions[env_ids, 1] = ys
        # 4) Update sphere poses:
        for i, eid in enumerate(env_ids):
            tr = gymapi.Transform()
            tr.p = gymapi.Vec3(float(xs[i]), float(ys[i]), 0.1)
            self.gym.set_actor_transform(
                self.envs[eid], self.sphere_handles[eid], tr
            )

    def compute_observations(self):
        obs = super().compute_observations()
        # Append vector-to-goal to observations:
        root_pos = self.robots_root_pos[:, :2]  # (N,2)
        vec_to_goal = self.target_positions - root_pos
        return torch.cat([obs, vec_to_goal], dim=1)

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        # Goal rewards:
        dist = torch.norm(self.robots_root_pos[:, :2] - self.target_positions, dim=1)
        rew += torch.exp(-dist) * 5.0                               # shaped reward
        rew += (dist < 0.5).float() * 50.0                          # success bonus
        # Respawn on success:
        success = dist < 0.5
        if success.any():
            self.reset_idx(success.nonzero(as_tuple=False).flatten())
        return obs, rew, done, info
