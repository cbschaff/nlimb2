"""Environment Wrapper for normalizing observations."""
import torch
import gym

class RewardScaleWrapper(gym.Wrapper):
    """Reward normalization."""

    def __init__(self, env, scale):
        """Init."""
        super().__init__(env)
        self.scale = torch.tensor([scale], dtype=torch.float, device=self.env.device)

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.env.step(action)
        return obs, rews * self.scale, dones, infos

    def reset(self, env_ids=None):
        return self.env.reset(env_ids)
