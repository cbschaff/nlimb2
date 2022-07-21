import dataclasses
from typing import Sequence
import random
from isaacgym import gymapi
import isaacgym.terrain_utils as tu
import numpy as np
import gin
import torch
from gym.spaces import Box
from pytorch3d import transforms


@dataclasses.dataclass
class Terrain():
    num_envs: int
    spacing: Sequence[float]
    num_per_row: int
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    x_border: float = 2.0
    y_border: float = 20.0

    def create(self, gym: gymapi.Gym, sim: gymapi.Sim):
        raise NotImplementedError

    def get_heights(self, points: torch.Tensor):
        """ points is an (... x 2) tensor """
        raise NotImplementedError

    def get_init_bounds(self):
        num_per_col = int(np.ceil(self.num_envs / self.num_per_row))
        low = -np.array(self.spacing)
        high = np.array([
                2 * self.spacing[0] * self.num_per_row - self.spacing[0],
                2 * self.spacing[1] * num_per_col - self.spacing[1],
                self.spacing[2]
            ])
        low[0] -= self.x_border
        low[1] -= self.y_border
        high[0] += self.x_border
        high[1] += self.y_border
        return low, high

    def get_terrain_bounds(self):
        raise NotImplementedError


@dataclasses.dataclass
class HeightFieldTerrain(Terrain):
    horizontal_scale: float = 0.1 # meters
    vertical_scale: float = 0.005 # meters

    def _get_terrain(self, num_rows, num_cols):
        return tu.SubTerrain(width=num_rows, length=num_cols,
                             vertical_scale=self.vertical_scale,
                             horizontal_scale=self.horizontal_scale)

    def _convert_heightfield_to_mesh(self, heightfield):
        vertices, triangles = tu.convert_heightfield_to_trimesh(
                heightfield, self.horizontal_scale, self.vertical_scale,
                slope_threshold=0.0)
        low, _ = self.get_init_bounds()
        params = gymapi.TriangleMeshParams()
        params.static_friction = self.static_friction
        params.dynamic_friction = self.dynamic_friction
        params.restitution = self.restitution
        params.nb_vertices = vertices.shape[0]
        params.nb_triangles = triangles.shape[0]
        params.transform.p.x = low[0]
        params.transform.p.y = low[1]

        return vertices, triangles, params

    def set_heightfield(self, gym: gymapi.Gym, sim: gymapi.Sim, heightfield: np.ndarray):
        self.heights = torch.from_numpy(heightfield).float() * self.vertical_scale
        vertices, triangles, params = self._convert_heightfield_to_mesh(heightfield)
        self.low = torch.tensor([params.transform.p.x, params.transform.p.y])
        self.high = self.low + (torch.tensor(self.heights.shape) - 1) * self.horizontal_scale
        self.size = self.high - self.low
        self.shape = torch.tensor(self.heights.shape)
        self.params = params
        gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), params)

    def get_heights(self, points: torch.Tensor):
        if not hasattr(self, 'heights'):
            raise RuntimeError('Call set_heightfield before get_heights')
        if points.device != self.heights.device:
            self.heights = self.heights.to(points.device)
            self.low = self.low.to(points.device)
            self.high = self.high.to(points.device)
            self.size = self.size.to(points.device)
            self.shape = self.shape.to(points.device)
        inds = ((points.clamp(self.low, self.high) - self.low) / self.size * self.shape).long()
        inds = torch.clamp(inds, max=self.shape - 2)
        h1 = self.heights[inds[..., 0], inds[..., 1]]
        h2 = self.heights[inds[..., 0]+1, inds[..., 1]+1]
        # The min is needed here because of how isaacgym.terrain_utils handles slope thresholds
        return torch.min(h1, h2)

    def get_terrain_bounds(self):
        if not hasattr(self, 'heights'):
            return None
        x, y = self.shape
        low_x = self.params.transform.p.x
        low_y = self.params.transform.p.y
        high_x = low_x + x * self.horizontal_scale
        high_y = low_y + y * self.horizontal_scale
        return (low_x, high_x), (low_y, high_y)

    def create(self, gym: gymapi.Gym, sim: gymapi.Sim):
        self.set_heightfield(gym, sim, self.build_heightfield())

    def build_heightfield(self):
        raise NotImplementedError


@gin.configurable(module='terrain')
class TerrainIndexer():
    def __init__(self, gym: gymapi.Gym, sim: gymapi.Sim, device: torch.device, terrain: Terrain,
                 actor_inds: torch.Tensor,
                 x_pos = [-.8, -.7, -.6, -.5, -.4, -.3, -.2, .2, .3, .4, .5, .6, .7, .8],
                 y_pos = [-.5, -.4, -.3, -.2, -.1, .1, .2, .3, .4, .5]):
        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = gym.get_env_count(sim)
        self.terrain = terrain
        self.actor_inds = actor_inds
        x_pos = torch.tensor(x_pos, device=device)
        y_pos = torch.tensor(y_pos, device=device)
        grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
        self.num_height_points = grid_x.numel()
        self.points = torch.zeros((self.num_envs, grid_x.numel(), 3), device=device)
        self.points[..., 0] = grid_x.flatten()
        self.points[..., 1] = grid_y.flatten()
        origins = [
            gym.get_env_origin(gym.get_env(sim, i)) for i in range(self.num_envs)
        ] # origins of each env
        self.origins = torch.tensor([[o.x, o.y] for o in origins], device=device)
        self._quat = torch.zeros((self.num_envs, 4), device=device)
        self.obs_space = Box(-np.inf, np.inf, (self.points.shape[1],))

    def __call__(self, tensor_api):
        # get yaw rotation
        self._quat[..., 0:1] = tensor_api.actor_root_state.orientation[self.actor_inds, 3:4]
        self._quat[..., 3:4] = tensor_api.actor_root_state.orientation[self.actor_inds, 2:3]
        self._quat /= torch.norm(self._quat, p=2, dim=-1, keepdim=True)
        # apply yaw rotation
        points = transforms.quaternion_apply(self._quat.unsqueeze(1), self.points)[..., :2]
        # translate into world frame
        points += tensor_api.actor_root_state.position.unsqueeze(1)[self.actor_inds, :, :2]
        points += self.origins.unsqueeze(1)
        # get heights
        return self.terrain.get_heights(points)



@gin.configurable(module='terrain')
@dataclasses.dataclass
class FlatTerrain(Terrain):
    def create(self, gym: gymapi.Gym, sim: gymapi.Sim):
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.plane_params.static_friction = self.static_friction
        self.plane_params.dynamic_friction = self.dynamic_friction
        self.plane_params.restitution = self.restitution
        gym.add_ground(sim, self.plane_params)

    def get_heights(self, points):
        return torch.zeros_like(points[..., 0])

    def get_terrain_bounds(self):
        return None


@gin.configurable(module='terrain')
@dataclasses.dataclass
class StairsTerrain(HeightFieldTerrain):
    length: float = 50.0 # meters
    stair_height: float = 0.1 # meters
    stair_width: float = 0.5 # meters

    def build_heightfield(self):
        low, high = self.get_init_bounds()
        size = (high - low)
        num_stair_rows = int(self.length // self.horizontal_scale)
        num_flat_rows = int(size[0] // self.horizontal_scale)
        num_cols = int(size[1] // self.horizontal_scale)
        heightfield = np.zeros((num_stair_rows + num_flat_rows, num_cols), dtype=np.int16)

        stairs = self._get_terrain(num_stair_rows, num_cols)
        heightfield[num_flat_rows:, :] = tu.stairs_terrain(
                stairs,
                step_width=self.stair_width,
                step_height=self.stair_height
        ).height_field_raw
        return heightfield


@gin.configurable(module='terrain')
@dataclasses.dataclass
class RandomStairsTerrain(HeightFieldTerrain):
    length: float = 50.0 # meters
    stair_height_min: float = 0.05 # meters
    stair_height_max: float = 0.2 # meters
    stair_width_min: float = 0.5 # meters
    stair_width_max: float = 1.0 # meters
    ramp_up_steps: int = 10
    up: bool = True
    include_starting_area: bool = True
    initial_height: float = 0.0

    def build_heightfield(self):
        low, high = self.get_init_bounds()
        size = (high - low)

        step_hmax = int(self.stair_height_max // self.vertical_scale)
        step_hmin = int(self.stair_height_min // self.vertical_scale)
        step_wmax = int(self.stair_width_max // self.horizontal_scale)
        step_wmin = int(self.stair_width_min // self.horizontal_scale)

        num_stair_rows = int(self.length // self.horizontal_scale)
        if self.include_starting_area:
            num_flat_rows = int(size[0] // self.horizontal_scale)
        else:
            num_flat_rows = 0
        num_cols = int(size[1] // self.horizontal_scale)
        heightfield = np.zeros((num_stair_rows + num_flat_rows, num_cols), dtype=np.int16)


        rows_used = 0
        steps = 0
        height = self.initial_height
        while rows_used < num_stair_rows:
            step_width = min(num_stair_rows - rows_used, random.randint(step_wmin, step_wmax))
            step_width = max(step_width, 2)
            hmin = int(min(1, (steps / self.ramp_up_steps)) * step_hmin)
            hmax = int(min(1, (steps / self.ramp_up_steps)) * step_hmax)
            step_height = random.randint(hmin, hmax)
            if self.up:
                height += step_height
            else:
                height -= step_height
            heightfield[num_flat_rows + rows_used:num_flat_rows + rows_used + step_width] = height
            rows_used += step_width
            steps += 1
        return heightfield


@gin.configurable(module='terrain')
@dataclasses.dataclass
class RandomWallsTerrain(HeightFieldTerrain):
    length: float = 50.0 # meters
    wall_height_min: float = 0.5 # meters
    wall_height_max: float = 1.0 # meters
    wall_distance_min: float = 2.0 # meters
    wall_distance_max: float = 3.0 # meters
    wall_thickness: float = 0.5 # meters
    ramp_up_steps: int = 10
    include_starting_area: bool = True
    initial_height: float = 0.0

    def build_heightfield(self):
        low, high = self.get_init_bounds()
        size = (high - low)

        wall_hmax = int(self.wall_height_max // self.vertical_scale)
        wall_hmin = int(self.wall_height_min // self.vertical_scale)
        wall_dmax = int(self.wall_distance_max // self.horizontal_scale)
        wall_dmin = int(self.wall_distance_min // self.horizontal_scale)
        wall_thickness = int(self.wall_thickness // self.horizontal_scale)

        num_wall_rows = int(self.length // self.horizontal_scale)
        if self.include_starting_area:
            num_flat_rows = int(size[0] // self.horizontal_scale)
        else:
            num_flat_rows = 0
        num_cols = int(size[1] // self.horizontal_scale)
        heightfield = self.initial_height * np.ones((num_wall_rows + num_flat_rows, num_cols), dtype=np.int16)

        rows_used = 0
        steps = 0
        while rows_used < num_wall_rows:
            wall_dist = random.randint(wall_dmin, wall_dmax)
            rows_used += wall_dist
            if rows_used >= num_wall_rows:
                break
            hmin = int(min(1, (steps / self.ramp_up_steps)) * wall_hmin)
            hmax = int(min(1, (steps / self.ramp_up_steps)) * wall_hmax)
            height = random.randint(hmin, hmax)

            heightfield[num_flat_rows + rows_used:num_flat_rows + rows_used + wall_thickness] = self.initial_height + height
            rows_used += wall_thickness
            steps += 1
        return heightfield


@gin.configurable(module='terrain')
@dataclasses.dataclass
class RandomGapsTerrain(HeightFieldTerrain):
    length: float = 50.0 # meters
    gap_min: float = 0.25 # meters
    gap_max: float = 1.0 # meters
    gap_distance_min: float = 1.0 # meters
    gap_distance_max: float = 3.0 # meters
    gap_height: float = -3.0 # meters
    ramp_up_steps: int = 5
    include_starting_area: bool = True
    initial_height: float = 0.0

    def build_heightfield(self):
        low, high = self.get_init_bounds()
        size = (high - low)

        gap_max = int(self.gap_max // self.horizontal_scale)
        gap_min = int(self.gap_min // self.horizontal_scale)
        gap_dmax = int(self.gap_distance_max // self.horizontal_scale)
        gap_dmin = int(self.gap_distance_min // self.horizontal_scale)
        gap_height = int(self.gap_height // self.vertical_scale)

        num_gap_rows = int(self.length // self.horizontal_scale)
        if self.include_starting_area:
            num_flat_rows = int(size[0] // self.horizontal_scale)
        else:
            num_flat_rows = 0
        num_cols = int(size[1] // self.horizontal_scale)
        heightfield = self.initial_height * np.ones((num_gap_rows + num_flat_rows, num_cols), dtype=np.int16)


        rows_used = 0
        steps = 0
        while rows_used < num_gap_rows:
            gap_dist = random.randint(gap_dmin, gap_dmax)
            rows_used += gap_dist
            if rows_used >= num_gap_rows:
                break
            wmin = int(min(1, (steps / self.ramp_up_steps)) * gap_min)
            wmax = int(min(1, (steps / self.ramp_up_steps)) * gap_max)
            gap_w = random.randint(wmin, wmax)
            heightfield[num_flat_rows + rows_used:num_flat_rows + rows_used + gap_w] = gap_height
            rows_used += gap_w
            steps += 1

        return heightfield


@gin.configurable(module='terrain')
@dataclasses.dataclass
class RandomTerrain(HeightFieldTerrain):
    length: float = 20.0 # meters

    def build_heightfield(self):
        terrains = [RandomStairsTerrain, RandomStairsTerrain,
                    RandomGapsTerrain, RandomWallsTerrain]
        random.shuffle(terrains)
        up_stairs = True
        heightfields = []
        initial_height = 0
        for i, terr in enumerate(terrains):
            if terr == RandomStairsTerrain:
                t = terr(num_envs=self.num_envs,
                         spacing=self.spacing,
                         num_per_row=self.num_per_row,
                         length=self.length,
                         up=up_stairs,
                         include_starting_area=i==0,
                         initial_height=initial_height)
                heightfields.append(t.build_heightfield())
                initial_height = heightfields[-1][-1, 0]
                up_stairs = False
            else:
                t = terr(num_envs=self.num_envs,
                         spacing=self.spacing,
                         num_per_row=self.num_per_row,
                         length=self.length,
                         include_starting_area=i==0,
                         initial_height=initial_height)
                heightfields.append(t.build_heightfield())

        return np.concatenate(heightfields, axis=0)
