from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from typing import Optional


class DOFStateTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        self._tensor_gym = self.gym.acquire_dof_state_tensor(self.sim)
        if num_envs is None:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym).view(-1, 2)
        else:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym).view(num_envs, -1, 2)
        self.refresh()
        self._last_refresh = self.gym.get_sim_time(self.sim)

    def _maybe_refresh(self):
        t = self.gym.get_sim_time(self.sim)
        if t > self._last_refresh:
            self.refresh()
        self._last_refresh = t

    @property
    def position(self):
        self._maybe_refresh()
        return self._tensor[..., 0]

    @position.setter
    def position(self, pos: torch.Tensor):
        self._tensor[..., 0] = pos

    @property
    def velocity(self):
        self._maybe_refresh()
        return self._tensor[..., 1]

    @velocity.setter
    def velocity(self, vel: torch.Tensor):
        self._tensor[..., 1] = vel

    @property
    def state(self):
        self._maybe_refresh()
        return self._tensor

    @state.setter
    def state(self, state: torch.Tensor):
        self._tensor[:] = state

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)

    def push(self, inds: Optional[torch.Tensor] = None):
        if inds is not None:
            if not isinstance(inds, torch.IntTensor):
                inds = inds.int()
            assert len(inds.shape) == 1
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                self._tensor_gym,
                gymtorch.unwrap_tensor(inds),
                len(inds)
            )
        else:
            self.gym.set_dof_state_tensor(self.sim, self._tensor_gym)
