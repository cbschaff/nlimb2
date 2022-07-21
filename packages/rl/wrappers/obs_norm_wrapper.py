"""Environment Wrapper for normalizing observations."""
from utils import nest, misc
from rl.modules import NestedRunningNorm
import gym
import torch
from typing import Tuple


def _mean(x):
    if x.dtype == torch.float or x.dtype == torch.double:
        return x.mean(dim=0)
    return None


def _var(x):
    if x.dtype == torch.float or x.dtype == torch.double:
        return x.var(dim=0)
    return None


def _clamp(x):
    if x.dtype == torch.float or x.dtype == torch.double:
        return x.clamp(-5.0, 5.0)
    return x


class ObsNormWrapper(gym.Wrapper):
    """Observation normalization for vecorized environments.

    Maintains a running norm of observations.
    """

    def __init__(self, env, update_norm_on_step=True):
        """Init."""
        super().__init__(env)
        self.running_norm = NestedRunningNorm(self.observation_space).to(self.env.device)
        self.update_norm_on_step = update_norm_on_step
        self._eval = False
        self._norm_finalized = False

    def update_obs_norm_with_random_transitions(self, steps):
        obs = self.env.reset()
        self.update_obs_norm(obs)
        for _ in range(steps):
            obs, _, _, _ = self.env.step(self.env.sample_actions())
            self.update_obs_norm(obs)

    def normalize(self, obs):
        """normalize."""
        if self.update_norm_on_step and not self._eval and not self._norm_finalized:
            self.update_obs_norm(obs)
        obs = self.running_norm(obs)
        return nest.map_structure(_clamp, obs)

    def update_obs_norm(self, obs):
        if self._norm_finalized:
            raise RuntimeError("Attempted to update obs norm after it has been finalized")
        mean = nest.map_structure(_mean, obs)
        var = nest.map_structure(_var, obs)
        count = nest.map_structure(lambda x: x.shape[0], obs)
        self.running_norm.update(mean, var, count)

    def unnorm_obs(self, obs):
        return self.running_norm(obs, unnorm=True)

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables norm updates.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode updates the norm every step.
        """
        self._eval = False

    def state_dict(self):
        """State dict."""
        return {'norm': self.running_norm.state_dict(),
                'finalized': self._norm_finalized}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.running_norm.load_state_dict(state_dict['norm'])
        self._norm_finalized = state_dict['finalized']

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.env.step(action)
        return self.normalize(obs), rews, dones, infos

    def finalize_obs_norm(self):
        self._norm_finalized = True

    def reset(self, env_ids=None):
        """Reset."""
        obs = self.env.reset(env_ids)
        return self.normalize(obs)
