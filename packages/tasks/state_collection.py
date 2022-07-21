from scenes import MixedXMLScene
from isaacgym import gymapi
import numpy as np
import torch
from xml_parser.geoms import Sphere, Capsule
import pytorch3d.transforms as transform
import gym


class TransformGatherer():
    def __init__(self, scene):
        self.api = scene.tensor_api
        self.nbodies = self.api.rigid_body_state.state.shape[0]
        self.device = scene.device
        self.actor_inds = torch.zeros(self.nbodies, device=self.device, dtype=torch.long)
        for i, env in enumerate(scene.envs):
            count = scene.gym.get_actor_rigid_body_count(env, 0)
            for j in range(count):
                ind = scene.gym.get_actor_rigid_body_index(env, 0, j, gymapi.IndexDomain.DOMAIN_SIM)
                self.actor_inds[ind] = i

        self.root_quat = torch.zeros((self.nbodies, 4), device=self.device, dtype=torch.float)
        self.body_quat = torch.zeros((self.nbodies, 4), device=self.device, dtype=torch.float)

    def get_frames(self, relative_to_root=True):
        body_pos = self.api.rigid_body_state.position
        body_quat = self.api.rigid_body_state.orientation
        self.body_quat[:, 1:] = body_quat[:, :3]
        self.body_quat[:, 0:1] = body_quat[:, 3:4]
        root_pos = self.api.actor_root_state.position[self.actor_inds]
        if relative_to_root:
            root_quat = self.api.actor_root_state.orientation[self.actor_inds]
            self.root_quat[:, 1:] = root_quat[:, :3]
            self.root_quat[:, 0:1] = root_quat[:, 3:4]
            root_quat_inv = transform.quaternion_invert(self.root_quat)

            body_quat = transform.quaternion_multiply(root_quat_inv, self.body_quat)
            body_pos = transform.quaternion_apply(root_quat_inv, body_pos - root_pos)

        else:
            body_pos = body_pos - root_pos
            body_quat = self.body_quat.clone()
        return body_pos, body_quat

    def get_absolute_frames(self):
        body_pos = self.api.rigid_body_state.position
        body_quat = self.api.rigid_body_state.orientation
        return body_pos.clone(), body_quat.clone()


class GeomObservation():
    def __init__(self, n_points: int):
        self.n_points = n_points
        self.points = []
        self.data = []
        self.env_ids = []
        self.frame_ids = []
        self.global_frame_ids = []
        self.geom_ids = []
        self._obs = None
        self._mask = None
        self._inds = None
        self.device = None
        self.num_envs = None
        self.max_bodies = None
        self.max_geoms = None

    def add_geom(self, env_id: int, frame_id: int, global_frame_id: int, geom_id: int,
                 points: torch.Tensor, data: torch.Tensor):
        if len(points) != self.n_points:
            raise ValueError(f'All geoms must have the same number of pionts ({self.n_points}). '
                             f'Found {len(points)} points.')

        self.points.append(points)
        self.env_ids.append(env_id)
        self.frame_ids.append(frame_id)
        self.global_frame_ids.append(global_frame_id)
        self.geom_ids.append(geom_id)
        self.data.append(data)

    def finalize(self, num_envs: int, max_bodies: int, device='cpu'):
        if len(self.points) == 0:
            return
        self.points = torch.stack(self.points).float().to(device)
        self.env_ids = torch.tensor(self.env_ids, dtype=torch.long, device=device)
        self.frame_ids = torch.tensor(self.frame_ids, dtype=torch.long, device=device)
        self.global_frame_ids = torch.tensor(self.global_frame_ids, dtype=torch.long, device=device)
        self.geom_ids = torch.tensor(self.geom_ids, dtype=torch.long, device=device)
        self.data = torch.stack(self.data).float().to(device)
        self.device = device
        self.num_envs = num_envs
        self.max_bodies = max_bodies
        self.max_geoms = torch.max(self.geom_ids) + 1
        self.n_obs = self.n_points * self.points.shape[-1] + self.data.shape[-1]

        self._obs = torch.zeros(
                (self.num_envs, self.max_bodies, self.max_geoms, self.n_obs),
                dtype=torch.float,
                device=self.device
        )
        self._mask = torch.zeros(
                (self.num_envs, self.max_bodies, self.max_geoms, 1),
                dtype=torch.bool,
                device=self.device
        )
        self._inds = (self.env_ids * self.max_bodies * self.max_geoms
                      + self.frame_ids * self.max_geoms + self.geom_ids)
        for i in range(self.num_envs):
            env_mask = self.env_ids == i
            for j in torch.unique(self.frame_ids[env_mask]):
                geom_mask = torch.logical_and(env_mask, self.frame_ids == j)
                n_geoms = torch.sum(geom_mask)
                self._mask[i, j, :n_geoms] = True

    def generate_obs(self, pos: torch.Tensor = None, ori: torch.Tensor = None):
        points = self.points
        if ori is not None:
            points = transform.quaternion_apply(ori[self.global_frame_ids[:, None]], points)
        if pos is not None:
            points = points + pos[self.global_frame_ids[:, None]]
        points = points.view(len(points), -1)

        obs = torch.cat([points, self.data], dim=1)
        self._obs.view(-1, obs.shape[1])[self._inds] = obs
        return {
            'obs': self._obs,
            'mask': self._mask,
        }

    def get_obs_space(self):
        return {'obs': gym.spaces.Box(-np.inf, np.inf, (self.n_obs,)),
                'mask': gym.spaces.Discrete(2)}


class JointObservation():
    def __init__(self):
        self.pos = []
        self.axis = []
        self.data = []
        self.frame_ids = []
        self.global_frame_ids = []
        self.env_ids = []
        self.joint_ids = []
        self.dof_ids = []
        self.parent_ids = []
        self._obs = None
        self._mask = None
        self._parents = None
        self._children = None
        self._inds = None
        self.device = None
        self.num_envs = None
        self.max_bodies = None
        self.max_joints = None

    def add_joint(self, env_id: int, frame_id: int, global_frame_id: int, joint_id: int,
                  dof_id: int, parent_id: int, pos: torch.Tensor, axis: torch.Tensor,
                  data: torch.Tensor):
        self.pos.append(pos)
        self.axis.append(axis)
        self.frame_ids.append(frame_id)
        self.global_frame_ids.append(global_frame_id)
        self.env_ids.append(env_id)
        self.joint_ids.append(joint_id)
        self.dof_ids.append(dof_id)
        self.parent_ids.append(parent_id)
        self.data.append(data)

    def finalize(self, num_envs, max_bodies: int, device='cpu'):
        self.pos = torch.stack(self.pos).float().to(device)
        self.axis = torch.stack(self.axis).float().to(device)
        self.axis.divide_(torch.linalg.norm(self.axis, dim=1, keepdims=True))
        self.frame_ids = torch.tensor(self.frame_ids, dtype=torch.long, device=device)
        self.global_frame_ids = torch.tensor(self.global_frame_ids, dtype=torch.long, device=device)
        self.env_ids = torch.tensor(self.env_ids, dtype=torch.long, device=device)
        self.joint_ids = torch.tensor(self.joint_ids, dtype=torch.long, device=device)
        self.dof_ids = torch.tensor(self.dof_ids, dtype=torch.long, device=device)
        self.parent_ids = torch.tensor(self.parent_ids, dtype=torch.long, device=device)
        self.data = torch.stack(self.data).float().to(device)
        self.device = device
        self.num_envs = num_envs
        self.max_bodies = max_bodies
        self.max_joints = torch.max(self.joint_ids) + 1
        self.n_obs = 9 + self.data.shape[-1]

        self._obs = torch.zeros(
                (self.num_envs, self.max_bodies, self.max_joints, self.n_obs),
                dtype=torch.float,
                device=self.device
        )
        self._mask = torch.zeros(
                (self.num_envs, self.max_bodies, self.max_joints, 1),
                dtype=torch.bool,
                device=self.device
        )
        self._parents = torch.zeros(
                (self.num_envs, self.max_bodies),
                dtype=torch.long,
                device=self.device
        )
        self._children = torch.zeros(
                (self.num_envs, self.max_bodies),
                dtype=torch.long,
                device=self.device
        )
        self._inds = (self.env_ids * self.max_bodies * self.max_joints
                      + self.frame_ids * self.max_joints + self.joint_ids)

        body_id_order = torch.arange(0, self.max_bodies)

        for i in range(self.num_envs):
            env_mask = self.env_ids == i
            for j in torch.unique(self.frame_ids[env_mask]):
                joint_mask = torch.logical_and(env_mask, self.frame_ids == j)
                n_joints = torch.sum(joint_mask)
                self._mask[i, j, :n_joints] = True
                self._parents[i, j] = self.parent_ids[joint_mask][0]
            self._children[i] = body_id_order

    def get_dof_state(self, dofs):
        self._obs.view(-1, self.n_obs)[self._inds][..., :2] = dofs[self.dof_ids]
        return self._obs[..., :2]

    def generate_obs(self, dofs: torch.Tensor, dof_forces: torch.tensor, pos: torch.Tensor = None,
                     ori: torch.Tensor = None):
        points = self.pos
        axis = self.axis
        if ori is not None:
            points = transform.quaternion_apply(ori[self.global_frame_ids], points)
            axis = transform.quaternion_apply(ori[self.global_frame_ids], axis)
        if pos is not None:
            points = points + pos[self.global_frame_ids]

        obs = torch.cat([dofs[self.dof_ids], dof_forces[self.dof_ids], points, axis, self.data],
                         dim=1)
        self._obs.view(-1, obs.shape[1])[self._inds] = obs
        return {
            'obs': self._obs,
            'mask': self._mask,
            'parents': self._parents,
            'children': self._children
        }

    def get_obs_space(self):
        return {'obs': gym.spaces.Box(-np.inf, np.inf, (self.n_obs,)),
                'mask': gym.spaces.Discrete(2),
                'parents': gym.spaces.Discrete(self.max_bodies),
                'children': gym.spaces.Discrete(self.max_bodies)}


class InertialObservation():
    def __init__(self):
        self.com = []
        self.mass = []
        self.inv_mass = []
        self.inertia = []
        self.frame_ids = []
        self.global_frame_ids = []
        self.env_ids = []
        self._obs = None
        self._mask = None
        self._inds = None
        self.device = None
        self.num_envs = None
        self.max_bodies = None

    def add_frame(self, env_id: int, frame_id: int, global_frame_id: int,
                  props: gymapi.RigidBodyProperties):
        self.com.append([props.com.x, props.com.y, props.com.z])
        self.mass.append(props.mass)
        self.inv_mass.append(props.invMass)
        I = props.inertia
        self.inertia.append([
            [I.x.x, I.x.y, I.x.z],
            [I.y.x, I.y.y, I.y.z],
            [I.z.x, I.z.y, I.z.z],
        ])
        self.frame_ids.append(frame_id)
        self.global_frame_ids.append(global_frame_id)
        self.env_ids.append(env_id)

    def finalize(self, num_envs, max_bodies: int, device='cpu'):
        self.com = torch.tensor(self.com).float().to(device)
        self.mass = torch.tensor(self.mass).float().to(device)
        self.inv_mass = torch.tensor(self.inv_mass).float().to(device)
        self.inertia = torch.tensor(self.inertia).float().to(device)
        self.env_ids = torch.tensor(self.env_ids, dtype=torch.long, device=device)
        self.frame_ids = torch.tensor(self.frame_ids, dtype=torch.long, device=device)
        self.global_frame_ids = torch.tensor(self.global_frame_ids, dtype=torch.long, device=device)
        self.device = device
        self.num_envs = num_envs
        self.max_bodies = max_bodies
        self.n_obs = 20

        self._obs = torch.zeros(
                (self.num_envs, self.max_bodies, self.n_obs),
                dtype=torch.float,
                device=self.device
        )
        self._mask = torch.zeros(
                (self.num_envs, self.max_bodies, 1),
                dtype=torch.bool,
                device=self.device
        )
        self._inds = self.env_ids * self.max_bodies + self.frame_ids
        for i in range(self.num_envs):
            n_frames = torch.sum(self.env_ids == i)
            self._mask[i, :n_frames] = True

    def generate_obs(self, force_sensor: torch.Tensor, pos: torch.Tensor = None,
                     ori: torch.Tensor = None):
        com = self.com
        I = self.inertia
        if ori is not None:
            com = transform.quaternion_apply(ori[self.global_frame_ids], com)
            ori = ori[self.global_frame_ids].unsqueeze(1)
            I = transform.quaternion_apply(ori, I.transpose(1,2))
            I = transform.quaternion_apply(ori, I.transpose(1,2))
        if pos is not None:
            com = com + pos[self.global_frame_ids]

        obs = torch.cat([force_sensor, com, I.flatten(1,2), self.mass[:, None],
                         self.inv_mass[:, None]], dim=1)
        self._obs.view(-1, obs.shape[1])[self._inds] = obs
        return {
            'obs': self._obs,
            'mask': self._mask
        }

    def get_obs_space(self):
        return {'obs': gym.spaces.Box(-np.inf, np.inf, (self.n_obs,)),
                'mask': gym.spaces.Discrete(2)}


class StateCollection():
    def __init__(self, scene):

        def _dfs(body, fn, data):
            fn(body, data)
            for b in body.bodies:
                _dfs(b, fn, data)
            return data

        self.max_bodies = 0
        self.max_geoms = 0
        self.max_joints = 0

        self.scene = scene
        self.api = scene.tensor_api
        self.geoms = GeomObservation(n_points=2)
        self.joints = JointObservation()
        self.inertia = InertialObservation()
        for i, (xml, env) in enumerate(zip(scene.xml_models, scene.envs)):
            bodies = _dfs(xml.root, lambda x, data: data.update({x.name: x}), {})
            parents = _dfs(xml.root,
                            lambda x, data: data.update({c.name: x.name for c in x.bodies}), {})
            parents[xml.root.name] = xml.root.name
            body_dict = scene.gym.get_actor_rigid_body_dict(env, 0)
            global_body_dict = {
                    ind: scene.gym.get_actor_rigid_body_index(env, 0, ind,
                                                               gymapi.IndexDomain.DOMAIN_SIM)
                    for _, ind in body_dict.items()
            }
            dof_dict = scene.gym.get_actor_dof_dict(env, 0)

            # collect inertial info
            props = scene.gym.get_actor_rigid_body_properties(env, 0)
            for j, p in enumerate(props):
                self.inertia.add_frame(i, j, global_body_dict[j], p)
            dof_inds = {}
            for body in bodies.values():
                for joint in body.joints:
                    dof_inds[joint.name] = scene.gym.get_actor_dof_index(
                            env, 0, dof_dict[joint.name], gymapi.IndexDomain.DOMAIN_SIM)

            self.max_bodies = max(self.max_bodies, len(bodies))
            for body in bodies.values():
                body_ind = body_dict[body.name]
                global_body_ind = global_body_dict[body_ind]
                parent_ind = body_dict[parents[body.name]]
                # collect geometric info
                self.max_geoms = max(self.max_geoms, len(body.geoms))
                for geom_ind, geom in enumerate(body.geoms):
                    if isinstance(geom, Capsule):
                        self.geoms.add_geom(
                                env_id=i,
                                frame_id=body_ind,
                                global_frame_id=global_body_ind,
                                geom_id=geom_ind,
                                points=torch.tensor([geom.p1, geom.p2]),
                                data=torch.tensor([geom.size, geom.density, *geom.friction])
                        )
                    if isinstance(geom, Sphere):
                        self.geoms.add_geom(
                                env_id=i,
                                frame_id=body_ind,
                                global_frame_id=global_body_ind,
                                geom_id=geom_ind,
                                points=torch.tensor([geom.pos, geom.pos]),
                                data=torch.tensor([geom.size, geom.density, *geom.friction])
                        )
                # collect joint info
                self.max_joints = max(self.max_joints, len(body.joints))
                for joint_ind, joint in enumerate(body.joints):
                    limits = [l * np.pi / 180. for l in joint.limits]
                    self.joints.add_joint(
                            env_id=i,
                            frame_id=body_ind,
                            global_frame_id=global_body_ind,
                            joint_id=joint_ind,
                            dof_id=dof_inds[joint.name],
                            parent_id=parent_ind,
                            pos=torch.tensor(joint.pos),
                            axis=torch.tensor(joint.axis),
                            data=torch.tensor([joint.stiffness, joint.gear, joint.armature, *limits])
                    )
        self.geoms.finalize(self.scene.num_envs, self.max_bodies, self.scene.device)
        self.joints.finalize(self.scene.num_envs, self.max_bodies, self.scene.device)
        self.inertia.finalize(self.scene.num_envs, self.max_bodies, self.scene.device)
        self.tg = TransformGatherer(scene)
        self._last_t = None
        self._obs = None

    def get_body_frames(self, relative_to_root=False):
        return self.tg.get_frames(relative_to_root)

    def get_absolute_body_frames(self):
        return self.tg.get_absolute_frames()

    def get_geom_obs(self, pos: torch.Tensor=None, quat: torch.Tensor=None):
        return self.geoms.generate_obs(pos, quat)

    def get_joint_obs(self, pos: torch.Tensor=None, quat: torch.Tensor=None):
        obs = self.joints.generate_obs(self.api.dof_state.state,
                                       self.api.dof_force.force.unsqueeze(1), pos, quat)
        return obs

    def get_inertial_obs(self, pos: torch.Tensor=None, quat: torch.Tensor=None):
        return self.inertia.generate_obs(self.api.force_sensor.twist, pos, quat)

    def get_root_obs(self, remove_xy=False):
        if remove_xy:
            return {'obs': self.api.actor_root_state.state[self.scene.actor_inds, 2:].clone()}
        else:
            return {'obs': self.api.actor_root_state.state[self.scene.actor_inds].clone()}

    def get_obs(self, ori_relative_to_root=False):
        # if self._need_to_compute_obs():
        pos, quat = self.get_body_frames(relative_to_root=ori_relative_to_root)
        self._obs = {}
        self._obs['geom'] = self.get_geom_obs(pos, quat)
        self._obs['inertia'] = self.get_inertial_obs(pos, quat)
        self._obs['joint'] = self.get_joint_obs(pos, quat)
        self._obs['root'] = self.get_root_obs(remove_xy=True)
        return self._obs

    def _need_to_compute_obs(self):
        t = self.scene.gym.get_sim_time(self.scene.sim)
        if self._last_t is None or t > self._last_t:
            self._last_t = t
            return True
        return False

    def get_obs_space(self):
        obs = {
            'geom': self.geoms.get_obs_space(),
            'inertia': self.inertia.get_obs_space(),
            'joint': self.joints.get_obs_space(),
            'root': {'obs': gym.spaces.Box(-np.inf, np.inf, shape=(11,))}
        }
        return obs

    def get_dof_state(self):
        return self.joints.get_dof_state(self.api.dof_state.state)


if __name__ == '__main__':
    scene = MixedXMLScene(num_envs=16, homogeneous_envs=False)
    scene.set_asset_root('../test/xmls')
    scene.create_scene()
    scene.reset()

    state = StateCollection(scene)

    scene.reset()
    for _ in range(1):
        scene.step()
        print(scene.tensor_api.actor_root_state.position[0])
        print(scene.tensor_api.actor_root_state.orientation[0])


    pos, quat = state.get_body_frames(relative_to_root=True)
    print(pos[state.tg.actor_inds == 0])
    print(quat[state.tg.actor_inds == 0])


    geom_obs = state.get_geom_obs(pos, quat)
    print(geom_obs['obs'].shape)
    print(geom_obs['mask'].shape)

    joint_obs = state.get_joint_obs(pos, quat)
    print(joint_obs['obs'].shape)
    print(joint_obs['mask'].shape)
    print(joint_obs['parents'].shape)

    inertia_obs = state.get_inertial_obs(pos, quat)
    print(inertia_obs['obs'].shape)
    print(inertia_obs['mask'].shape)

    root_obs = state.get_root_obs()
    print(root_obs['obs'].shape)
    print(root_obs['obs'][0])

    print(state.get_obs_space())

    for i in range(10):
        import time
        t = time.time()
        state.get_obs()
        print(time.time() - t)
        if i % 3 == 0:
            scene.step()
