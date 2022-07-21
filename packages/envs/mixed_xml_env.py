from typing import Optional
from isaacgym import gymapi
from envs.isaac_env import IsaacEnv
from scenes import MixedXMLScene
import torch
import gin


@gin.configurable(module='envs')
class IsaacMixedXMLEnv(IsaacEnv):
    def __init__(self, task, num_envs: int, device: str = 'cuda:0', enable_graphics: bool = False,
                 create_eval_sensors: bool = False, asset_root: str ='/xmls',
                 homogeneous_envs: bool = True, spacing: list = [5., 5., 5.],
                 num_per_row=None, terrain=None, zero_reset=False):

        self.scene = MixedXMLScene(num_envs, device=device,
                                   create_eval_sensors=create_eval_sensors,
                                   homogeneous_envs=homogeneous_envs,
                                   spacing=spacing,
                                   num_per_row=num_per_row,
                                   terrain=terrain,
                                   zero_reset=zero_reset)
        self.asset_root = asset_root
        self.scene.set_asset_root(self.asset_root)
        self.init_scene()
        self.task = task(self.scene)
        self.homogeneous_envs = homogeneous_envs
        self.enable_graphics = enable_graphics
        if not self.homogeneous_envs:
            self._inds = self.task.state_collection.joints._inds
        IsaacEnv.__init__(self, self.scene, self.task, enable_graphics)

    def init_scene(self):
        if self.scene.sim is not None:
            self.scene.destroy_scene()
        self.scene.create_scene()

    def step(self, action: torch.Tensor, realtime=False):
        if not self.homogeneous_envs:
            action = action.view(-1)[self._inds]
        return IsaacEnv.step(self, action, realtime=realtime)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if self.scene.sim is None:
            self.init_scene()
        return IsaacEnv.reset(self, env_ids)

    def record_env(self, env_id: int, outfile: str):
        if env_id >= len(self.scene.cameras):
            raise ValueError(f"Env {env_id} does not have a camera")
        self.scene.record_camera(f'env{env_id}', self.scene.cameras[env_id],
                                 gymapi.IMAGE_COLOR, outfile)

    def write_env_video(self, env_id: int):
        self.scene.write_camera_video(f'env{env_id}')

    def get_asset_root(self):
        return self.asset_root

    def update_designs(self, designs):
        self.task.update_designs(designs)
