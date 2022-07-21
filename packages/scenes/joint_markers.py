from isaacgym import gymapi
import torch
from scipy.spatial.transform import Rotation
import numpy as np
import pytorch3d.transforms as transforms
from utils import Transform


class JointMarkers():
    def __init__(self, gym, sim, env, env_id, actor, xml_model):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.env_id = env_id
        self.actor = actor
        self.xml = xml_model
        self._sphere = self.gym.create_sphere(self.sim, 0.05)
        self._xbox = self.gym.create_box(self.sim, 0.12, 0.08, 0.08)
        self._ybox = self.gym.create_box(self.sim, 0.08, 0.12, 0.08)
        self._zbox = self.gym.create_box(self.sim, 0.08, 0.08, 0.12)
        self._actor_inds = None
        self._create_markers()

    def _create_markers(self):

        self.marker_dict = {}

        def _dfs(body, fn, data):
            fn(body, data)
            for b in body.bodies:
                _dfs(b, fn, data)
            return data

        self.bodies = _dfs(self.xml.root, lambda x, data: data.update({x.name: x}), {})
        self.parents = _dfs(self.xml.root,
                            lambda x, data: data.update({c.name: x.name for c in x.bodies}), {})
        self.parents[self.xml.root.name] = self.xml.root.name
        pose = gymapi.Transform()
        for body in self.bodies.values():
            if len(body.joints) == 3:
                marker = self.gym.create_actor(self.env, self._sphere, pose,
                                               f'{body.name}_marker', self.env_id, 1, 0)
                self.gym.set_rigid_body_color(self.env, marker, 0, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(*(1.0, 1.0, 1.0)))
                self.marker_dict[body.name] = marker

            elif len(body.joints) == 0:
                continue
            else:
                axis = body.joints[0].axis
                if abs(axis[0]) > 0.5:
                    asset = self._xbox
                elif abs(axis[1]) > 0.5:
                    asset = self._ybox
                else:
                    asset = self._zbox

                marker = self.gym.create_actor(self.env, asset, pose,
                                               f'{body.name}_marker', self.env_id, 1, 0)
                if body.joints[0].limits[1] == 0.:  # knee joint
                    self.gym.set_rigid_body_color(self.env, marker, 0, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(*(0.7*190./255., 0.7*174./255., 212./255.)))
                elif body.joints[0].limits[0] == 0.:  # elbow joint
                    self.gym.set_rigid_body_color(self.env, marker, 0, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(0.7*174./255., 212./255., 0.7*190./255.))

                else:
                    self.gym.set_rigid_body_color(self.env, marker, 0, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(*(abs(a) for a in body.joints[0].axis)))
                self.marker_dict[body.name] = marker

    def step(self, tensor_api):
        device = tensor_api.rigid_body_state.state.device
        if self._actor_inds is None:
            self._actor_inds = {}
            for marker in self.marker_dict.values():
                self._actor_inds[marker] = self.gym.get_actor_index(self.env, marker,
                                                                    gymapi.IndexDomain.DOMAIN_SIM)
                self._actor_inds_t = torch.tensor(list(self._actor_inds.values()),
                                                  device=device, dtype=torch.int32)

            self.body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.actor)
            self.global_body_dict = {
                    ind: self.gym.get_actor_rigid_body_index(self.env, 0, ind,
                                                             gymapi.IndexDomain.DOMAIN_SIM)
                    for _, ind in self.body_dict.items()
            }

        for body_name, marker in self.marker_dict.items():
            body_ind = self.global_body_dict[self.body_dict[body_name]]
            parent_ind = self.global_body_dict[self.body_dict[self.parents[body_name]]]
            actor_ind = self._actor_inds[marker]
            tensor_api.actor_root_state.state[actor_ind] = tensor_api.rigid_body_state.state[body_ind]

        tensor_api.actor_root_state.push(self._actor_inds_t)





