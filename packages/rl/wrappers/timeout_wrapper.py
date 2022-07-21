"""Implements environment timeouts."""
import gym
import torch


class TimeoutWrapper(gym.Wrapper):
    def __init__(self, env, timeout):
        gym.Wrapper.__init__(self, env)
        self.timeout = timeout
        self.count = torch.zeros((self.env.num_envs,), dtype=torch.int, device=self.env.device)

    def reset(self, env_ids=None):
        self.count[env_ids] = 0
        return self.env.reset(env_ids)

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.count += 1
        self.count[done] = 0
        timed_out = self.count >= self.timeout
        if torch.any(timed_out):
            inds = timed_out.nonzero().squeeze(dim=1)
            ob = self.reset(inds)
            done[inds] = True
        return ob, r, done, info


