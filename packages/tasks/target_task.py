from isaacgym import gymapi
from envs.isaac_env import IsaacEnv
from scenes import MixedXMLScene
from utils import torch_jit_utils as tjit
from tasks.state_collection import StateCollection
from tasks.task_common import TaskCommon
import torch
import gin
import gym
import numpy as np
from utils.misc import pack_gym_space
import pytorch3d.transforms as transforms
from envs.isaac_env import Task


def to_torch(array, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(array, dtype=dtype, device=device, requires_grad=requires_grad)


@gin.configurable(module='tasks')
class MoveToTarget(Task):
    def __init__(self, scene: MixedXMLScene,
                 target: list = (1000., 0., 0.),
                 heading_vec: list = (1., 0., 0.),
                 up_vec: list = (0., 0., 1.),
                 progress_weight: float = 1.0,
                 alive_reward: float = 0.5,
                 timeout: int = 1000,
                 termination_height: float = 0.31,
                 termination_cost: float = 2.0,
                 up_weight: float = 0.1,
                 heading_weight: float = 0.5,
                 actions_cost_scale: float = 0.005,
                 energy_cost_scale: float = 0.01,
                 joints_at_limit_cost_scale: float = 0.1,
                 progress_timeout: int = 100):
        self.scene = scene
        self.state_collection = StateCollection(scene)
        ob_space = self.state_collection.get_obs_space()
        ob_space['task'] = {'obs': gym.spaces.Box(-np.inf, np.inf, shape=(6,))}
        ob_space['terrain'] = {'obs': scene.terrain_indexer.obs_space}

        self.observation_space = pack_gym_space(ob_space)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.target = to_torch(target, device=scene.device).unsqueeze(0)
        self.dt = scene.sim_params.dt
        self.prev_potentials = to_torch([-1000./self.dt] * scene.num_envs, device=scene.device)

        self.up_vec = to_torch(up_vec, device=scene.device).unsqueeze(0)
        self.up_vec /= torch.norm(self.up_vec)

        self.heading_vec = to_torch(heading_vec, device=scene.device).unsqueeze(0)
        self.heading_vec /= torch.norm(self.heading_vec)

        self.progress_weight = progress_weight
        self.progress_timeout = progress_timeout
        self.up_weight = up_weight
        self.heading_weight = heading_weight
        self.actions_cost_scale = actions_cost_scale
        self.energy_cost_scale = energy_cost_scale
        self.joints_at_limit_cost_scale = joints_at_limit_cost_scale
        self.task_common = TaskCommon(
            alive_reward=alive_reward,
            termination_height=termination_height,
            termination_cost=termination_cost,
            up_weight=up_weight,
            heading_weight=heading_weight,
            actions_cost_scale=actions_cost_scale,
            energy_cost_scale=energy_cost_scale,
            joints_at_limit_cost_scale=joints_at_limit_cost_scale
        )

        self._task_obs = torch.zeros((scene.num_envs, 6), device=scene.device, dtype=torch.float)
        self._done = torch.zeros((scene.num_envs,), dtype=torch.bool, device=scene.device)
        self._reward = torch.zeros((scene.num_envs,), dtype=torch.float, device=scene.device)
        self._dof_inds = self.state_collection.joints._inds
        dof_mask = self.state_collection.joints._mask.squeeze(-1)
        self._actions = torch.zeros_like(dof_mask, dtype=torch.float)

        self._root_ori = None
        self._time_since_progress = torch.zeros((scene.num_envs,), dtype=torch.long,
                                                device=scene.device)
        self._best_potential = torch.empty((scene.num_envs,), dtype=torch.float,
                                           device=scene.device)
        self._best_potential.fill_(-torch.inf)

        self.timeout = timeout
        self._steps = torch.zeros((scene.num_envs,), dtype=torch.long, device=scene.device)

    def _update_root_ori(self, ori):
        if self._root_ori is None:
            self._root_ori = torch.zeros_like(ori)

        self._root_ori[:, 0:1] = ori[:, 3:4]
        self._root_ori[:, 1:] = ori[:, :3]

    def _compute_potential_and_target_dir(self, tensor_api):
        to_target = self.target - tensor_api.actor_root_state.position[self.scene.actor_inds]
        to_target[:, 2] = 0.0
        norm = torch.norm(to_target, p=2, dim=-1)
        potentials = -norm / self.dt
        target_dir = to_target / norm.unsqueeze(1)
        return potentials, target_dir

    def observe_fn(self, tensor_api, actions):
        obs = self.state_collection.get_obs()
        self._update_root_ori(tensor_api.actor_root_state.orientation[self.scene.actor_inds])
        root_orientation = transforms.quaternion_invert(self._root_ori)

        potentials, target_dir = self._compute_potential_and_target_dir(tensor_api)
        self.prev_potentials.copy_(potentials)

        target_dir_in_root_frame = transforms.quaternion_apply(root_orientation, target_dir)
        self._task_obs[:, :3] = target_dir_in_root_frame
        self._task_obs[:, 3:] = self.heading_vec
        obs['task'] = {'obs': self._task_obs}
        height = tensor_api.actor_root_state.position[self.scene.actor_inds, 2:3]
        obs['terrain'] = {'obs': self.scene.terrain_indexer(self.scene.tensor_api) - height}
        return obs

    def done_fn(self, tensor_api, actions):
        return self.task_common.compute_termination_common(
                        tensor_api.actor_root_state.position[self.scene.actor_inds])

    def reward_fn(self, tensor_api, actions):
        dof_state = self.state_collection.get_dof_state()
        self._update_root_ori(tensor_api.actor_root_state.orientation[self.scene.actor_inds])
        self._actions.view(-1)[self._dof_inds] = actions

        potentials, target_dir = self._compute_potential_and_target_dir(tensor_api)

        new_best = self._best_potential < potentials
        self._time_since_progress[new_best] = 0
        self._time_since_progress[torch.logical_not(new_best)] += 1
        self._best_potential[new_best] = potentials[new_best]
        self._steps += 1

        self._reward[:] = self.task_common.compute_reward_common(
            tensor_api.actor_root_state.position[self.scene.actor_inds],
            self._root_ori,
            actions=self._actions,
            dof_pos=dof_state[..., 0],
            dof_vel=dof_state[..., 1],
            up_vec=self.up_vec,
            heading_vec=self.heading_vec,
            target_vecs=target_dir
        )

        self._reward += self.progress_weight * (potentials - self.prev_potentials)
        return self._reward

    def reset(self, env_ids=None):
        if env_ids is None:
            self._time_since_progress.fill_(0)
            self._best_potential.fill_(-torch.inf)
            self._steps.fill_(0)
        else:
            self._time_since_progress[env_ids] = 0
            self._best_potential[env_ids] = -torch.inf
            self._steps[env_ids] = 0

    def termination_for_data_quality(self, tensor_api):
        """Have the algorithm silently terminate the episode if it has been a while since
        progress has been made, or we hit a maximum number of steps."""
        return torch.logical_or(
            self._time_since_progress >= self.progress_timeout,
            self._steps >= self.timeout
        )

    def update_designs(self, designs):
        pass


@gin.configurable(module='tasks')
class SwitchingTarget(MoveToTarget):
    def __init__(self, scene: MixedXMLScene,
                 initial_target = (10., 0., 0.),
                 min_target_dist = 3.0,
                 max_target_dist = 10.0,
                 target_reached_reward = 2.0,
                 distance_threshold = 0.2,
                 heading_vec: list = (1., 0., 0.),
                 up_vec: list = (0., 0., 1.),
                 progress_weight: float = 1.0,
                 alive_reward: float = 0.5,
                 timeout: int = 1000,
                 termination_height: float = 0.31,
                 termination_cost: float = 2.0,
                 up_weight: float = 0.1,
                 heading_weight: float = 0.5,
                 actions_cost_scale: float = 0.005,
                 energy_cost_scale: float = 0.01,
                 joints_at_limit_cost_scale: float = 0.1,
                 progress_timeout: int = 100):
        MoveToTarget.__init__(self, scene,
                              target=initial_target,
                              heading_vec=heading_vec,
                              up_vec=up_vec,
                              progress_weight=progress_weight,
                              alive_reward=alive_reward,
                              timeout=timeout,
                              termination_height=termination_height,
                              termination_cost=termination_cost,
                              up_weight=up_weight,
                              heading_weight=heading_weight,
                              actions_cost_scale=actions_cost_scale,
                              energy_cost_scale=energy_cost_scale,
                              joints_at_limit_cost_scale=joints_at_limit_cost_scale,
                              progress_timeout=progress_timeout)

        self.target_reached_reward = target_reached_reward
        self.min_target_dist = min_target_dist
        self.max_target_dist = max_target_dist
        self.distance_threshold = distance_threshold
        self.target = to_torch(initial_target,
                               device=scene.device).unsqueeze(0).repeat(scene.num_envs, 1)
        self.gen_target_sequence(initial_target)
        self._target_inds = torch.zeros(scene.num_envs, dtype=torch.long, device=scene.device)

    def gen_target_sequence(self, initial_target):
        n_targets = 20
        min_y = -5.0
        max_y = 5.0
        min_x = -1.0
        max_x = 5.0
        self._target_seq = torch.zeros((n_targets, 3))
        self._target_seq[0] = torch.tensor(initial_target)

        def _sample_target(prev_target):
            prev_x = prev_target[0]
            new_x = prev_x + torch.rand(1)*(max_x - min_x) + min_x
            new_y = torch.rand(1)*(max_y - min_y) + min_y
            return torch.cat([new_x, new_y, torch.zeros(1)], dim=0)

        for i in range(1, n_targets):
            new_target = _sample_target(self._target_seq[i-1])
            self._target_seq[i] = new_target
        self._target_seq = self._target_seq.to(self.scene.device)

    def target_reached(self, tensor_api):
        xy = tensor_api.actor_root_state.position[self.scene.actor_inds, :2]
        dists = torch.norm(xy - self.target[..., :2], dim=1)
        return dists <= self.distance_threshold

    def reward_fn(self, tensor_api, actions):
        MoveToTarget.reward_fn(self, tensor_api, actions)

        target_reached = self.target_reached(tensor_api)
        if torch.any(target_reached):
            self._reward += self.target_reached_reward * self.target_reached(tensor_api)
            self._target_inds[target_reached] += 1
            self.target.copy_(self._target_seq[self._target_inds])
            self._best_potential[target_reached] = -torch.inf

        return self._reward

    def reset(self, env_ids=None):
        MoveToTarget.reset(self, env_ids)
        if env_ids is None:
            self._target_inds.fill_(0)
            self.target.copy_(self._target_seq[self._target_inds])
        else:
            self._target_inds[env_ids] = 0
            self.target[env_ids].copy_(self._target_seq[self._target_inds[env_ids]])
