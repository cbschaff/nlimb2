from typing import Optional, List
from isaacgym import gymapi
from sim import CameraSensor
from .isaac_scene import IsaacScene
from utils import torch_jit_utils as tjit
import torch
import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
from xml_parser import XMLModel
from .terrain import TerrainIndexer
from .joint_markers import JointMarkers


def to_torch(array, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(array, dtype=dtype, device=device, requires_grad=requires_grad)

def get_init_z(paths):
    z = []
    for path in paths:
        tree = ET.parse(path)
        z.append(float(tree.getroot().find('worldbody').find('body').get('pos').split()[2]))
    return z

class MixedXMLScene(IsaacScene):

    def __init__(self, *args, **kwargs):
        IsaacScene.__init__(self, *args, **kwargs)
        self.asset_root = None
        self.load_urdf = False

    def set_asset_root(self, path):
        self.asset_root = path

    def set_viewer_pos(self, pos: list, target: list):
        scale = np.array([2 * self.spacing[0] * self.num_per_row,
                          2 * self.spacing[1] * self.num_envs / self.num_per_row,
                          self.spacing[2]])
        offset = np.array([self.spacing[0], self.spacing[1], 0.0])
        pos = np.array(pos) * scale - offset
        pos[0] -= 5.0
        pos[2] += 2.0
        target = np.array(target) * scale - offset
        IsaacScene.set_viewer_pos(self, pos.tolist(), target.tolist())

    def create_viewer(self, props: gymapi.CameraProperties = None):
        if self._viewer_pos is None:
            self.set_viewer_pos([0.5, 0.5, 1.0], [0.0, 0.5, 0.0])
        if props is None:
            props = gymapi.CameraProperties()
            props.height = 432 * 2
            props.width = 768 * 2
        IsaacScene.create_viewer(self, props)

    def act(self, action: torch.Tensor):
        self.tensor_api.dof_force_control.force = action * self.joint_gears
        self.tensor_api.dof_force_control.push()

    def create_scene(self):
        if self.asset_root is None:
            raise ValueError('You must set an asset root before calling create_scene')
        self.asset_files = sorted([os.path.basename(f) for f in
                                   glob.glob(os.path.join(self.asset_root, '*.xml'))])

        xml_models = []
        for i, asset in enumerate(self.asset_files):
            xml = XMLModel.from_path(os.path.join(self.asset_root, asset))
            xml_models.append(xml)
        self.xml_models = [xml_models[i % len(xml_models)] for i in range(self.num_envs)]

        self.sim = self.gym.create_sim(self.device_id, self.device_id, gymapi.SIM_PHYSX,
                                       self.sim_params)

        self.terrain.create(self.gym, self.sim)

        lower = gymapi.Vec3(-self.spacing[0], -self.spacing[1], 0.0)
        upper = gymapi.Vec3(*self.spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.angular_damping = 0.0

        start_pose = gymapi.Transform()
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z,
                                            start_pose.r.w],
                                           device=self.device)
        self.inv_start_rot = tjit.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.num_dof = []
        self.num_bodies = []
        self.envs = []
        self.joint_gears = []
        self.cameras = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.markers = []
        self.init_z = get_init_z([os.path.join(self.asset_root, f) for f in self.asset_files])
        self.xml_assets = [self.gym.load_asset(self.sim, self.asset_root, path, asset_options)
                           for path in self.asset_files]
        if self.load_urdf:
            asset_options.replace_cylinder_with_capsule = True
            asset_options.override_com = True
            asset_options.override_inertia = True
            self.asset_files = []
            for i, m in enumerate(xml_models):
                fname = f'{i:05d}.urdf'
                m.write_urdf(os.path.join(self.asset_root, fname))
                self.asset_files.append(fname)
            self.urdf_assets = [self.gym.load_asset(self.sim, self.asset_root, path, asset_options)
                                for path in self.asset_files]
        else:
            self.urdf_assets = self.xml_assets

        # add body force sensors
        if self.load_urdf:
            for asset in self.urdf_assets:
                num_bodies = self.gym.get_asset_rigid_body_count(asset)
                sensor_pose = gymapi.Transform()
                sensor_props = gymapi.ForceSensorProperties()
                sensor_props.enable_forward_dynamics_forces = True
                sensor_props.enable_constraint_solver_forces = True
                sensor_props.use_world_frame = True
                for body_idx in range(num_bodies):
                    ind = self.gym.create_asset_force_sensor(asset, body_idx, sensor_pose,
                                                             sensor_props)
        else:
            for asset in self.xml_assets:
                num_bodies = self.gym.get_asset_rigid_body_count(asset)
                sensor_pose = gymapi.Transform()
                sensor_props = gymapi.ForceSensorProperties()
                sensor_props.enable_forward_dynamics_forces = True
                sensor_props.enable_constraint_solver_forces = True
                sensor_props.use_world_frame = True
                for body_idx in range(num_bodies):
                    ind = self.gym.create_asset_force_sensor(asset, body_idx, sensor_pose,
                                                             sensor_props)

        for i in range(self.num_envs):
            xml = self.xml_assets[i % len(self.xml_assets)]
            urdf = self.urdf_assets[i % len(self.urdf_assets)]
            asset = urdf if self.load_urdf else xml
            start_pose = gymapi.Transform()
            start_pose.p.z = self.init_z[i % len(self.init_z)]
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, self.num_per_row
            )
            handle = self.gym.create_actor(env_ptr, asset, start_pose, "actor", i, 1, 0)
            if self.load_urdf:
                props = self.gym.get_asset_dof_properties(xml)
                self.gym.set_actor_dof_properties(env_ptr, handle, props)
                # props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
                # self.gym.set_actor_rigid_body_properties(env_ptr, handle, props)
                # props = self.gym.get_asset_rigid_shape_properties(xml)
                # self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)



            num_dof = self.gym.get_asset_dof_count(asset)
            num_bodies = self.gym.get_asset_rigid_body_count(asset)
            self.num_dof.append(num_dof)
            self.num_bodies.append(num_bodies)

            actuator_props = self.gym.get_asset_actuator_properties(xml)
            motor_efforts = [prop.motor_effort for prop in actuator_props]
            self.joint_gears.append(to_torch(motor_efforts, device=self.device))
            props = self.gym.get_actor_dof_properties(env_ptr, handle)
            for prop_ind, prop in enumerate(props):
                prop[4] = 2 * np.pi
                prop[5] = motor_efforts[prop_ind]
            self.gym.set_actor_dof_properties(env_ptr, handle, props)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)
            if self.create_eval_sensors:
                props = gymapi.CameraProperties()
                props.height = 1024
                props.width = 1024
                self.cameras.append(CameraSensor(self.gym, self.sim, env_ptr, props))
                t = gymapi.Transform()
                t.p = gymapi.Vec3(0, -3, 1)
                t.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90))
                body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, handle, 0)
                self.cameras[-1].attach_camera_to_body(body_handle, t, gymapi.FOLLOW_POSITION)

            if self.num_envs == 1:
                self.markers.append(JointMarkers(self.gym, self.sim, env_ptr, i, handle,
                                                 self.xml_models[i]))
                self._set_colors(self.xml_models[i], env_ptr, handle, default=True)
            else:
                self._set_colors(self.xml_models[i], env_ptr, handle)


            self.envs.append(env_ptr)
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
            dof_limits_lower = []
            dof_limits_upper = []
            for j in range(num_dof):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    dof_limits_lower.append(dof_prop['upper'][j])
                    dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    dof_limits_lower.append(dof_prop['lower'][j])
                    dof_limits_upper.append(dof_prop['upper'][j])

            self.dof_limits_lower.append(to_torch(dof_limits_lower, device=self.device))
            self.dof_limits_upper.append(to_torch(dof_limits_upper, device=self.device))
        self.joint_gears = torch.cat(self.joint_gears, dim=0)
        self.dof_limits_lower = torch.cat(self.dof_limits_lower, dim=0)
        self.dof_limits_upper = torch.cat(self.dof_limits_upper, dim=0)
        if self.homogeneous_envs:
            self.joint_gears = self.joint_gears.view(self.num_envs, -1)
            self.dof_limits_lower = self.dof_limits_lower.view(self.num_envs, -1)
            self.dof_limits_upper = self.dof_limits_upper.view(self.num_envs, -1)

        self.gym.prepare_sim(self.sim)

        self.initial_dof_pos = torch.zeros_like(self.tensor_api.dof_state.position, device=self.device)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor,
                                                       self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.tensor_api.dof_state.velocity)
        self.initial_root_state = self.tensor_api.actor_root_state.state.clone()
        self.initial_root_state[..., 7:13] = 0  # set lin_vel and ang_vel to 0

        self.env_dof_inds = [0] + np.cumsum(self.num_dof).tolist()


        self.actor_inds = torch.tensor(
                [self.gym.get_actor_index(env, 0, gymapi.IndexDomain.DOMAIN_SIM)
                 for env in self.envs],
                dtype=torch.long, device=self.device)
        self.terrain_indexer = TerrainIndexer(self.gym, self.sim, self.device, self.terrain,
                                              self.actor_inds)
        if self.zero_reset:
            self.initial_root_state[..., :2] = -self.terrain_indexer.origins


    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = tuple(range(self.num_envs))
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        elif env_ids.dtype is not torch.long:
            env_ids = env_ids.long()

        noise = torch.rand_like(self.initial_dof_pos) * 0.4 - 0.2
        init_pos = self.initial_dof_pos + noise
        torch.minimum(init_pos, self.dof_limits_upper, out=init_pos)
        torch.maximum(init_pos, self.dof_limits_lower, out=init_pos)
        init_vel = torch.rand_like(self.initial_dof_vel) * 0.2 - 0.1
        if self.homogeneous_envs:
            self.tensor_api.dof_state.position[env_ids] = init_pos[env_ids]
            self.tensor_api.dof_state.velocity[env_ids] = init_vel[env_ids]
        else:
            for env_id in env_ids:
                start, end = self.env_dof_inds[env_id], self.env_dof_inds[env_id + 1]
                self.tensor_api.dof_state.position[start:end] = init_pos[start:end]
                self.tensor_api.dof_state.velocity[start:end] = init_vel[start:end]
        actor_ids = self.actor_inds[env_ids]
        self.tensor_api.actor_root_state.state[actor_ids] = self.initial_root_state[actor_ids]

        env_ids_int32 = env_ids.int()
        self.tensor_api.dof_state.push(env_ids_int32)
        self.tensor_api.actor_root_state.push(env_ids_int32)

    def step(self, *args, **kwargs):
        def _callback():
            for marker in self.markers:
                marker.step(self.tensor_api)
        IsaacScene.step(self, *args, callback=_callback, **kwargs)

    def _set_colors(self, xml, env, actor, default=False):
        def _dfs(body, fn, data):
            fn(body, data)
            for b in body.bodies:
                _dfs(b, fn, data)
            return data

        body_dict = self.gym.get_actor_rigid_body_dict(env, actor)
        bodies = _dfs(xml.root, lambda x, data: data.update({x.name: x}), {})
        for body in bodies.values():
            body_ind = body_dict[body.name]
            if default:
                color = (0.97, 0.38, 0.06)
            elif len(body.joints) == 3:
                color = (1.0, 1.0, 1.0)
            elif len(body.joints) == 0:
                color = (0.0, 0.0, 0.0)
            else:
                color = tuple([abs(a) for a in body.joints[0].axis])
            self.gym.set_rigid_body_color(env, actor, body_ind, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(*color))

