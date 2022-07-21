from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from typing import Optional


class RigidBodyStateTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        self._tensor_gym = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if num_envs is None:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym).view(-1, 13)
        else:
            self._tensor = gymtorch.wrap_tensor(self._tensor_gym).view(num_envs, -1, 13)
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
        return self._tensor[..., 0:3]

    @position.setter
    def position(self, pos: torch.Tensor):
        self._tensor[..., 0:3] = pos

    @property
    def orientation(self):
        self._maybe_refresh()
        return self._tensor[..., 3:7]

    @orientation.setter
    def orientation(self, ori: torch.Tensor):
        self._tensor[..., 3:7] = ori

    @property
    def linvel(self):
        self._maybe_refresh()
        return self._tensor[..., 7:10]

    @linvel.setter
    def linvel(self, vel: torch.Tensor):
        self._tensor[..., 7:10] = vel

    @property
    def angvel(self):
        self._maybe_refresh()
        return self._tensor[..., 10:13]

    @angvel.setter
    def angvel(self, vel: torch.Tensor):
        self._tensor[..., 10:13] = vel

    @property
    def state(self):
        self._maybe_refresh()
        return self._tensor

    @state.setter
    def state(self, state: torch.Tensor):
        self._tensor[:] = state

    def refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def push(self):
        self.gym.set_rigid_body_state_tensor(self.sim, self._tensor_gym)
