from isaacgym import gymapi
from typing import Optional
from .actor_root_state_tensor import ActorRootStateTensor
from .force_sensor_tensor import ForceSensorTensor
from .dof_state_tensor import DOFStateTensor
from .dof_force_tensor import DOFForceTensor
from .rigid_body_state_tensor import RigidBodyStateTensor
from .control_tensor import DOFActuationForceTensor, DOFPositionTargetTensor, DOFVelocityTargetTensor


class TensorApi():
    """Utility class for viewing and updating info with the isaacgym tensor api."""

    def __init__(self,
                 gym: gymapi.Gym,
                 sim: gymapi.Sim,
                 device: str,
                 num_envs: Optional[int] = None,
                 ):
        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self._tensors = {}

    def _get_tensor(self, name, obj):
        if name not in self._tensors:
            self._tensors[name] = obj(self.gym, self.sim, self.device,
                                      self.num_envs)
        return self._tensors[name]

    @property
    def actor_root_state(self):
        return self._get_tensor('actor_root_state', ActorRootStateTensor)

    @property
    def rigid_body_state(self):
        return self._get_tensor('rigid_body_state', RigidBodyStateTensor)

    @property
    def force_sensor(self):
        return self._get_tensor('force_sensor', ForceSensorTensor)

    @property
    def dof_state(self):
        return self._get_tensor('dof_state', DOFStateTensor)

    @property
    def dof_force(self):
        return self._get_tensor('dof_force', DOFForceTensor)

    @property
    def dof_force_control(self):
        return self._get_tensor('dof_force_control', DOFActuationForceTensor)

    @property
    def dof_position_control(self):
        return self._get_tensor('dof_position_control', DOFPositionTargetTensor)

    @property
    def dof_velocity_control(self):
        return self._get_tensor('dof_velocity_control', DOFVelocityTargetTensor)
