from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from typing import Optional


class DOFActuationForceTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        if num_envs is None:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device)
        else:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device).view(num_envs, -1)
        self._tensor_gym = gymtorch.unwrap_tensor(self._tensor)

    @property
    def force(self):
        return self._tensor

    @force.setter
    def force(self, f: torch.Tensor):
        self._tensor[:] = f

    def push(self, inds: Optional[torch.Tensor] = None):
        if inds is not None:
            if not isinstance(inds, torch.IntTensor):
                inds = inds.int()
            assert len(inds.shape) == 1
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                self._tensor_gym,
                gymtorch.unwrap_tensor(inds),
                len(inds)
            )
        else:
            self.gym.set_dof_actuation_force_tensor(self.sim, self._tensor_gym)


class DOFPositionTargetTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        if num_envs is None:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device)
        else:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device).view(num_envs, -1)
        self._tensor_gym = gymtorch.unwrap_tensor(self._tensor)

    @property
    def target(self):
        return self._tensor

    @target.setter
    def target(self, f: torch.Tensor):
        self._tensor[:] = f

    def push(self, inds: Optional[torch.Tensor] = None):
        if inds is not None:
            if not isinstance(inds, torch.IntTensor):
                inds = inds.int()
            assert len(inds.shape) == 1
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                self._tensor_gym,
                gymtorch.unwrap_tensor(inds),
                len(inds)
            )
        else:
            self.gym.set_dof_position_target_tensor(self.sim, self._tensor_gym)


class DOFVelocityTargetTensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: str,
                 num_envs: Optional[int] = None):
        self.gym = gym
        self.sim = sim
        if num_envs is None:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device)
        else:
            self._tensor = torch.zeros(self.gym.get_sim_dof_count(self.sim),
                                       dtype=torch.float,
                                       device=device).view(num_envs, -1)
        self._tensor_gym = gymtorch.unwrap_tensor(self._tensor)

    @property
    def target(self):
        return self._tensor

    @target.setter
    def target(self, f: torch.Tensor):
        self._tensor[:] = f

    def push(self, inds: Optional[torch.Tensor] = None):
        if inds is not None:
            if not isinstance(inds, torch.IntTensor):
                inds = inds.int()
            assert len(inds.shape) == 1
            self.gym.set_dof_velocity_target_tensor_indexed(
                self.sim,
                self._tensor_gym,
                gymtorch.unwrap_tensor(inds),
                len(inds)
            )
        else:
            self.gym.set_dof_velocity_target_tensor(self.sim, self._tensor_gym)
