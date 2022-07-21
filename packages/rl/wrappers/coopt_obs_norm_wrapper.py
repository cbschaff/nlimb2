"""Environment Wrapper for normalizing observations."""
from utils import nest, misc
from rl.modules import NestedRunningNorm
import gym
import torch
from typing import Tuple


def _clamp(x):
    if x.dtype == torch.float or x.dtype == torch.double:
        return x.clamp(-5.0, 5.0)
    return x


class CoOptObsNormWrapper(gym.Wrapper):
    """Masked version of Observation normalization wrapper.

    This version is a little hacky, and only designed to work with the specific
    observation structure that I am using for this project.
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

        mean = nest.get_structure(obs)
        var = nest.get_structure(obs)
        count = nest.get_structure(obs)
        assert sorted(list(obs.keys())) == ['geom', 'inertia', 'joint', 'root', 'task', 'terrain']

        for k in ['geom', 'inertia', 'joint']:
            mask = obs[k]['mask'].squeeze(-1)
            mean[k]['obs'] = obs[k]['obs'][mask].mean(dim=0)
            var[k]['obs'] = obs[k]['obs'][mask].var(dim=0)
            count[k]['obs'] = mask.sum()

        for k in ['root', 'task', 'terrain']:
            mean[k]['obs'] = obs[k]['obs'].mean(dim=0)
            var[k]['obs'] = obs[k]['obs'].var(dim=0)
            count[k]['obs'] = obs[k]['obs'].shape[0]

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
