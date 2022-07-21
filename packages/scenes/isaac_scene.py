from typing import Optional, Tuple
import time
from isaacgym import gymapi
from sim import TensorApi, SimParams, CameraSensor, ViewerRecorder
from .terrain import FlatTerrain
import torch
import numpy as np


class IsaacScene():
    def __init__(self, num_envs: int, device: str = 'cuda:0',
                 homogeneous_envs: bool = True, create_eval_sensors: bool = False,
                 spacing=[5., 5., 5.], num_per_row=None, terrain=None, zero_reset=False):
        self.num_envs = num_envs
        if num_per_row is None:
            self.num_per_row = int(np.sqrt(num_envs))
        else:
            self.num_per_row = num_per_row
        self.device = device
        self.device_id = int(self.device.split(':')[-1])
        self.homogeneous_envs = homogeneous_envs
        self.create_eval_sensors = create_eval_sensors
        self.spacing = spacing
        if terrain is None:
            self.terrain = FlatTerrain(self.num_envs, self.spacing, self.num_per_row)
        else:
            self.terrain = terrain(self.num_envs, self.spacing, self.num_per_row)
        self.zero_reset = zero_reset
        self.sim_params = SimParams()
        self.gym = gymapi.acquire_gym()
        self.viewer = None
        self.sim = None
        self._gpu_image_tensors_locked = False
        self._tensor_api = None
        self._viewer_recorder = None
        self._camera_recorders = {}
        self._paused = False
        self.sim_time = 0.0
        self._viewer_pos = None
        self._viewer_target = None

    def create_scene(self):
        pass

    def act(self, action: torch.Tensor):
        pass

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        pass

    def set_viewer_pos(self, pos: list, target: list):
        self._viewer_pos = gymapi.Vec3(*pos)
        self._viewer_target = gymapi.Vec3(*target)
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(self.viewer, None, self._viewer_pos, self._viewer_target)

    def create_viewer(self, props: gymapi.CameraProperties = None):
        if props is None:
            props = gymapi.CameraProperties()
        if self.viewer is None:
            self.viewer = self.gym.create_viewer(self.sim, props)
            self.gym.viewer_camera_look_at(self.viewer, None, self._viewer_pos, self._viewer_target)
            self.create_viewer_events()

    def save_viewer_image(self, fname):
        if self.viewer is not None:
            self.gym.write_viewer_image_to_file(self.viewer, fname)

    @property
    def tensor_api(self):
        if self._tensor_api is None:
            if self.homogeneous_envs:
                self._tensor_api = TensorApi(self.gym, self.sim, self.device, self.num_envs)
            else:
                self._tensor_api = TensorApi(self.gym, self.sim, self.device)
        return self._tensor_api

    def step(self,
             step_graphics: bool = False,
             sync_frame_time: bool = False,
             lock_gpu_image_tensors: bool = False,
             callback: callable = None
             ):
        self.sim_time += self.sim_params.dt
        if self._gpu_image_tensors_locked:
            self.gym.end_access_image_tensors(self.sim)
            self._gpu_image_tensors_locked = False
        self.gym.simulate(self.sim)
        if callback is not None:
            callback()
        if self.device == 'cpu' or self.viewer is not None or len(self._camera_recorders) > 0:
            self.gym.fetch_results(self.sim, True)

        if step_graphics or self.viewer is not None or len(self._camera_recorders) > 0:
            self.gym.step_graphics(self.sim)
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
                if self._viewer_recorder is not None:
                    self._viewer_recorder.record_frame()
            if step_graphics or len(self._camera_recorders) > 0:
                self.gym.render_all_camera_sensors(self.sim)
                for recorders in self._camera_recorders.values():
                    recorders.record_frame()

        if self.viewer is not None:
            self.gym.poll_viewer_events(self.viewer)
            self.handle_viewer_events()

        if lock_gpu_image_tensors:
            self.gym.start_access_image_tensors(self.sim)
            self._gpu_image_tensors_locked = True

    def create_viewer_events(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "exit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "record_viewer")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "play_pause")

    def handle_viewer_events(self, paused=False):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "exit" and evt.value > 0:
                self.gym.destroy_viewer(self.viewer)
                self.viewer = None
                self._paused = False
                break

            if evt.action == "record_viewer" and evt.value > 0:
                if self._viewer_recorder is None:
                    print('Recording Viewer... Press "R" again to stop and create a video.')
                    self.record_viewer('./isaac_viewer.mp4')
                else:
                    print('Recording Viewer...')
                    self.write_viewer_video()
            elif evt.action == "play_pause" and evt.value > 0:
                self._paused = not self._paused
                if self._paused:
                    print("Simulation Paused... Press 'space' to unpause.")
                else:
                    print("Resuming Simulation...")
                while self._paused:
                    time.sleep(0.01)
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    self.gym.poll_viewer_events(self.viewer)
                    self.handle_viewer_events(paused)

    def record_viewer(self, outfile):
        if self.viewer is not None:
            self._viewer_recorder = ViewerRecorder(self.gym, self.sim, self.viewer, outfile)

    def write_viewer_video(self):
        if self._viewer_recorder is None:
            raise RuntimeError("Call 'record_viewer' before 'create_viewer_video'")
        self._viewer_recorder.write_video()
        self._viewer_recorder = None

    def record_camera(self, name: str, camera: CameraSensor, img_type: gymapi.ImageType,
                      outfile: str):
        self._camera_recorders[name] = camera.record(img_type, outfile)

    def write_camera_video(self, name: str):
        if name not in self._camera_recorders:
            raise ValueError(
                f"No recording with name {name}. Call 'record_camera' before 'create_camera_video'."
            )
        self._camera_recorders[name].write_video()
        del self._camera_recorders[name]

    def destroy_veiwer(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

    def destroy_scene(self):
        if self.sim is not None:
            self.destroy_veiwer()
            self.gym.destroy_sim(self.sim)
            self.sim = None
            self._tensor_api = None
