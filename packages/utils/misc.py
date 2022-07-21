"""Misc utilities."""
import numpy as np
from . import nest
import torch
from functools import partial
import gym


def _discount(x, gamma, not_done, end_value=None):
    n = x.shape[0]
    out = torch.zeros_like(x)
    out[-1] = x[-1]
    if end_value is not None:
        out[-1] += gamma * not_done[-1] * end_value
    for ind in reversed(range(n - 1)):
        out[ind] = x[ind] + not_done[ind] * gamma * out[ind + 1]
    return out


def discount(x, gamma, not_done, end_value=None):
    """Return the discounted sum of a sequence."""
    return nest.map_structure(partial(_discount, gamma=gamma, not_done=not_done,
                                      end_value=end_value), x)


def unpack_gym_space(space):
    """Change gym.spaces.Dict and gym.spaces.Tuple to be dictionaries and
       tuples."""
    if isinstance(space, gym.spaces.Dict):
        return {
            k: unpack_gym_space(v) for k, v in space.spaces.items()
        }
    elif isinstance(space, gym.spaces.Tuple):
        return [unpack_gym_space(space) for space in space.spaces]
    else:
        return space


def pack_gym_space(space):
    """Change nested dictionaries and tuples of gym.spaces.Space objects to
       gym.spaces.Dict and gym.spaces.Tuple."""
    if isinstance(space, dict):
        return gym.spaces.Dict({
            k: pack_gym_space(v) for k, v in space.items()
        })
    elif isinstance(space, (list, tuple)):
        return gym.spaces.Tuple([pack_gym_space(s) for s in space])
    else:
        return space


def set_env_to_eval_mode(env):
    """Set env and all wrappers to eval mode if available."""
    if hasattr(env, 'eval'):
        env.eval()
    if hasattr(env, 'venv'):
        set_env_to_eval_mode(env.venv)
    elif hasattr(env, 'env'):
        set_env_to_eval_mode(env.env)


def set_env_to_train_mode(env):
    """Set env and all wrappers to train mode if available."""
    if hasattr(env, 'train'):
        env.train()
    if hasattr(env, 'venv'):
        set_env_to_train_mode(env.venv)
    elif hasattr(env, 'env'):
        set_env_to_train_mode(env.env)


def env_state_dict(env):
    def _env_state_dict(env, state_dict, ind):
        """Gather the state of env and all its wrappers into one dict."""
        if hasattr(env, 'state_dict'):
            state_dict[ind] = env.state_dict()
        if hasattr(env, 'venv'):
            state_dict = _env_state_dict(env.venv, state_dict, ind+1)
        elif hasattr(env, 'env'):
            state_dict = _env_state_dict(env.env, state_dict, ind+1)
        return state_dict
    return _env_state_dict(env, {}, 0)


def env_load_state_dict(env, state_dict, ind=0):
    """Load the state of env and its wrapprs."""
    if hasattr(env, 'load_state_dict'):
        env.load_state_dict(state_dict[ind])
    if hasattr(env, 'venv'):
        env_load_state_dict(env.venv, state_dict, ind+1)
    elif hasattr(env, 'env'):
        env_load_state_dict(env.env, state_dict, ind+1)


class ActionBounds():
    def __init__(self, action_space, device):
        self.action_space = unpack_gym_space(action_space)
        self.device = device
        self.get_bounds()

    def get_bounds(self):
        def _get_lower(space):
            if isinstance(space, gym.spaces.Box):
                return torch.from_numpy(space.low).to(self.device)
            else:
                return None

        def _get_upper(space):
            if isinstance(space, gym.spaces.Box):
                return torch.from_numpy(space.high).to(self.device)
            else:
                return None

        self.lower = nest.map_structure(_get_lower, self.action_space)
        self.upper = nest.map_structure(_get_upper, self.action_space)

    def _clamp(self, item):
        action, lower, upper = item
        if lower is not None:
            return torch.minimum(torch.maximum(action, lower), upper)
        else:
            return action

    def clamp(self, actions):
        return nest.map_structure(self._clamp, nest.zip_structure(actions, self.lower, self.upper))

    def _norm(self, item):
        action, lower, upper = item
        if lower is not None:
            return 2.0 * (action - lower) / (upper - lower) - 1.0
        else:
            return action

    def norm(self, actions):
        return nest.map_structure(self._norm, nest.zip_structure(actions, self.lower, self.upper))

    def _unnorm(self, item):
        action, lower, upper = item
        if lower is not None:
            return 0.5 * (action + 1) * (upper - lower) + lower
        else:
            return action

    def unnorm(self, actions):
        return nest.map_structure(self._unnorm, nest.zip_structure(actions, self.lower, self.upper))

    def sample(self, num_envs):
        def _sample(item):
            lower, upper, space = item
            if lower is None:
                actions = [space.sample() for _ in range(num_envs)]
                return torch.from_numpy(np.asarray(actions)).to(self.device)
            else:
                a = torch.rand([num_envs, 3], device=self.device)
                actions = torch.rand([num_envs] + list(lower.shape), device=self.device)
                return actions * (upper - lower) + lower
        return nest.map_structure(_sample, nest.zip_structure(self.lower, self.upper,
                                                              self.action_space))

    def zero_action(self, num_envs):
        return nest.map_structure(
            lambda x: torch.zeros((num_envs, *x.shape), dtype=x.dtype, device=x.device),
            self.lower
        )
