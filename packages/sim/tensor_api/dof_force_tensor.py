from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from typing import Optional


class DOFForceTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        self._tensor_gym = self.gym.acquire_dof_force_tensor(self.sim)
        if num_envs is None:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym)
        else:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym).view(num_envs, -1)
        self.refresh()
        self._last_refresh = self.gym.get_sim_time(self.sim)

    def _maybe_refresh(self):
        t = self.gym.get_sim_time(self.sim)
        if t > self._last_refresh:
            self.refresh()
        self._last_refresh = t

    @property
    def force(self):
        self._maybe_refresh()
        return self._tensor

    @force.setter
    def force(self, f: torch.Tensor):
        raise RuntimeError("DOF Force Tensor is read only!")

    def refresh(self):
        self.gym.refresh_dof_force_tensor(self.sim)
