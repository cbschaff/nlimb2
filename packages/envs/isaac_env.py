from typing import Optional
from scenes import IsaacScene
import gym
import torch
from utils.misc import ActionBounds


class Task():
    def observe_fn(self, tensor_api, action):
        raise NotImplementedError

    def reward_fn(self, tensor_api, action):
        raise NotImplementedError

    def done_fn(self, tensor_api, action):
        raise NotImplementedError

    def reset(self, env_ids):
        raise NotImplementedError

    def termination_for_data_quality(self, tensor_api, action):
        raise NotImplementedError


class IsaacEnv():
    """Base Class for Environments implemented with IsaacGym."""
    def __init__(self, scene: IsaacScene, task: Task, enable_graphics: bool = False):

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.scene = scene
        self.task = task
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space
        self.num_envs = self.scene.num_envs
        self.device = scene.device
        self.enable_graphics = enable_graphics
        self.action_bounds = ActionBounds(self.action_space, self.device)

    def step(self, action: torch.Tensor, realtime=False):
        assert isinstance(action, torch.Tensor)
        action = self.action_bounds.clamp(action)
        self.scene.act(action)
        self.scene.step(step_graphics=self.enable_graphics,
                        lock_gpu_image_tensors=self.enable_graphics, sync_frame_time=realtime)
        done = self.task.done_fn(self.scene.tensor_api, action)
        reward = self.task.reward_fn(self.scene.tensor_api, action)
        env_ids = done.nonzero()
        if len(env_ids) > 0:
            self.scene.reset(env_ids[:, 0])
            self.task.reset(env_ids[:, 0])
            action[env_ids[:, 0]] = 0
        obs = self.task.observe_fn(self.scene.tensor_api, action)
        return obs, reward, done, {}

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.scene.reset()
        else:
            self.scene.reset(env_ids)
        self.task.reset(env_ids)
        return self.task.observe_fn(self.scene.tensor_api,
                                    self.action_bounds.zero_action(self.scene.num_envs))

    def get_data_quality_terminations(self):
        return self.task.termination_for_data_quality(self.scene.tensor_api)

    def set_viewer_pos(self, pos: list, target: list):
        self.scene.set_viewer_pos(pos, target)

    def get_num_envs(self):
        return self.num_envs

    def get_device(self):
        return self.device

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def create_viewer(self):
        self.scene.create_viewer()

    def destroy_viewer(self):
        self.scene.destroy_viewer()

    def sample_actions(self):
        return self.action_bounds.sample(self.num_envs)

    def record_viewer(self, outfile: str):
        if self.scene.viewer is None:
            self.scene.create_viewer()
        self.scene.record_viewer(outfile)

    def save_viewer_image(self, fname):
        self.scene.save_viewer_image(fname)

    def write_viewer_video(self):
        self.scene.write_viewer_video()

    def record_env(self, env_id: int, outfile: str):
        raise NotImplementedError

    def write_env_video(self, env_id: int):
        raise NotImplementedError

    def close(self):
        self.scene.destroy_scene()
