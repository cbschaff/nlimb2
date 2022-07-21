"""Environment Wrapper for logging episode data."""
import gym
import torch
import wandb
import numpy as np
import time
from rl.modules import RunningNorm


class EpisodeLogger(gym.Wrapper):
    """Logs episode data."""

    def __init__(self, env, namespace=''):
        """Init."""
        super().__init__(env)
        self.t = 0
        self.rews = torch.zeros(self.num_envs, dtype=torch.float, device=self.env.device)
        self.lens = torch.zeros(self.num_envs, dtype=torch.float, device=self.env.device)
        self.batch_size = self.env.num_envs
        self.rew_rn = RunningNorm((1,)).to(self.env.device)
        self.len_rn = RunningNorm((1,)).to(self.env.device)
        self.max_rew = -np.inf
        self._eval = False
        self.ns = 'episode/'
        if len(namespace) > 0:
            self.ns += namespace + '_'
        wandb.define_metric('episode/*', summary='max', step_metric="train/step")
        self._time_of_last_log = time.time()
        self._best_reward = -np.inf

    def reset(self, env_ids=None):
        """Reset."""
        obs = self.env.reset(env_ids)
        if not self._eval and env_ids is not None:
            self._update_rn(self.rew_rn, self.rews[env_ids])
            self._update_rn(self.len_rn, self.lens[env_ids])
            self.max_rew = max(self.max_rew, torch.max(self.rews[env_ids]))
        if not self._eval and self.rew_rn.count >= self.batch_size:
            self.log()
        self.rews[env_ids] = 0.
        self.lens[env_ids] = 0
        return obs

    def log(self):
        if time.time() - self._time_of_last_log > 1.0:
            data = {}
            data[f'{self.ns}length'] = self.len_rn.mean.cpu().numpy()
            data[f'{self.ns}length_std'] = self.len_rn.std.cpu().numpy()
            data[f'{self.ns}reward'] = self.rew_rn.mean.cpu().numpy()
            data[f'{self.ns}reward_std'] = self.rew_rn.std.cpu().numpy()
            data[f'{self.ns}max_reward'] = self.max_rew
            self._best_reward = max(self._best_reward, data[f'{self.ns}reward'])
            data[f'{self.ns}best_reward'] = self._best_reward
            wandb.log(data)
            self._time_of_last_log = time.time()
            self.len_rn.reset()
            self.rew_rn.reset()
            self.max_rew = -np.inf

    def _update_rn(self, rn, data):
        count = len(data)
        mean = data.mean()
        if count == 1:
            rn.update(mean, torch.zeros_like(mean), count)
        else:
            rn.update(mean, data.var(), count)

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.env.step(action)
        if not self._eval:
            self.t += self.num_envs
            self.rews += rews
            self.lens += 1
            done_inds = dones.nonzero().squeeze(dim=1)
            count = len(done_inds)
            if count > 0:
                self._update_rn(self.rew_rn, self.rews[done_inds])
                self._update_rn(self.len_rn, self.lens[done_inds])
                self.max_rew = max(self.max_rew, torch.max(self.rews[done_inds]))

            if self.rew_rn.count >= self.batch_size:
                self.log()

            self.rews[done_inds] = 0.
            self.lens[done_inds] = 0
        return obs, rews, dones, infos

    @property
    def best_reward(self):
        return self._best_reward

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs episode stats.
        """
        self._eval = False

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']
