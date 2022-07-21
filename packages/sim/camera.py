from isaacgym import gymapi
from isaacgym import gymtorch
import tempfile
import imageio


class CameraRecorder():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, env: gymapi.Env, camera: int,
                 camera_props: gymapi.CameraProperties, img_type: gymapi.ImageType,
                 outfile: str):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.camera = camera
        self.img_type = img_type
        self.props = camera_props
        self.writer = imageio.get_writer(outfile, format='FFMPEG', mode='I',
                                         fps=1. / gym.get_sim_params(sim).dt,
                                         quality=10)

    def record_frame(self):
        if self.writer is None:
            raise ValueError('Tried to record after the video has been writen')
        image = self.gym.get_camera_image(self.sim, self.env, self.camera, self.img_type)
        if len(image.shape) == 2:
            image = image.reshape((image.shape[0], -1, 4))
        self.writer.append_data(image[..., :-1])

    def write_video(self):
        self.writer.close()
        self.writer = None

    def __del__(self):
        if self.writer is not None:
            self.writer.close()


class ViewerRecorder():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, viewer: gymapi.Viewer, outfile: str):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.writer = imageio.get_writer(outfile, format='FFMPEG', mode='I',
                                         fps=1. / gym.get_sim_params(sim).dt)
        self.tempfile = tempfile.NamedTemporaryFile(suffix='.png')

    def record_frame(self):
        if self.writer is None:
            raise ValueError('Tried to record after the video has been writen')
        # currently, the only way to get the viewer image is to write it to a file
        self.gym.write_viewer_image_to_file(self.viewer, self.tempfile.name)
        img = imageio.imread(self.tempfile.name)
        self.writer.append_data(img)

    def write_video(self):
        self.writer.close()
        self.tempfile.close()
        self.writer = None

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
            self.tempfile.close()


class CameraSensor():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, env: gymapi.Env,
                 camera_props: gymapi.CameraProperties):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.props = camera_props
        self.camera = self.gym.create_camera_sensor(self.env, self.props)

        # gpu api
        self._rgb_tensor = None
        self.rgb_tensor = None
        self._depth_tensor = None
        self.depth_tensor = None

    def init_rgb_tensor(self):
        if not self.props.enable_tensors:
            raise ValueError('You must set "CameraProperties.enable_tensors = True" to create gpu tensors')

        self._rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera,
                                                                gymapi.IMAGE_COLOR)
        self.rgb_tensor = gymtorch.wrap_tensor(self._rgb_tensor)

    def init_depth_tensor(self):
        if not self.props.enable_tensors:
            raise ValueError('You must set "CameraProperties.enable_tensors = True" to create gpu tensors')

        self._depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera,
                                                                  gymapi.IMAGE_DEPTH)
        self.depth_tensor = gymtorch.wrap_tensor(self._depth_tensor)

    def set_camera_location(self, pos: gymapi.Vec3, target: gymapi.Vec3):
        self.gym.set_camera_location(self.camera, self.env, pos, target)

    def set_camera_transform(self, t: gymapi.Transform):
        self.gym.set_camera_transform(self.camera, self.env, t)

    def attach_camera_to_body(self, body: int, t: gymapi.Transform, mode: gymapi.CameraFollowMode):
        self.gym.attach_camera_to_body(self.camera, self.env, body, t, mode)

    def save_image(self, file: str, img_type: gymapi.ImageType):
        self.gym.write_camera_image_to_file(self.sim, self.env, self.camera, img_type, file)

    def get_image(self, img_type: gymapi.ImageType):
        return self.gym.get_camera_image(self.sim, self.env, self.camera, img_type)

    def record(self, img_type: gymapi.ImageType, outfile: str):
        return CameraRecorder(self.gym, self.sim, self.env, self.camera, self.props, img_type,
                              outfile)
