import torch
import pytorch3d.transforms as transforms
from dataclasses import dataclass

@dataclass
class TaskCommon():
    alive_reward: float = 0.5
    termination_height: float = 0.31
    termination_cost: float = 2.0
    up_weight: float = 0.1
    heading_weight: float = 0.1
    actions_cost_scale: float = 0.005
    energy_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1

    def compute_reward_common(self,
            root_position: torch.tensor,    # nenv, 3
            root_orientation: torch.tensor, # nenv, 4
            actions: torch.tensor,          # nenv, max_bodies, max_dofs
            dof_pos: torch.tensor,          # nenv, max_bodies, max_dofs
            dof_vel: torch.tensor,          # nenv, max_bodies, max_dofs
            up_vec: torch.tensor,           # nenv, 3
            heading_vec: torch.tensor,      # nenv, 3
            target_vecs: torch.tensor,      # nenv, 3
        ):
        n = root_position.shape[0]

        # reward for keeping the robot upright
        up_vec_in_base_frame = transforms.quaternion_apply(root_orientation, up_vec)
        up_reward = self.up_weight * up_vec_in_base_frame[..., 2]
        up_reward = torch.clamp(up_reward, min=0.)

        # reward for keeping the robot pointed at its target
        heading_vec_in_base_frame = transforms.quaternion_apply(root_orientation, heading_vec)
        heading_reward = self.heading_weight * (heading_vec_in_base_frame * target_vecs).sum(-1)
        heading_reward = torch.clamp(heading_reward, min=0.)

        # energy penalty for movement
        actions_cost = (actions ** 2).sum(dim=(1, 2))
        electricity_cost = (torch.abs(actions * dof_vel)).sum(dim=(1,2))
        dof_at_limit_cost = torch.sum(torch.abs(dof_pos) > 0.99, dim=(1,2)).float()

        total_reward = (self.alive_reward + up_reward + heading_reward
                - self.actions_cost_scale * actions_cost
                - self.energy_cost_scale * electricity_cost
                - self.joints_at_limit_cost_scale * dof_at_limit_cost)

        # adjust reward for fallen agents
        total_reward = torch.where(root_position[..., 2] < self.termination_height,
                                   -1 * torch.ones_like(total_reward) * self.termination_cost,
                                   total_reward)
        return total_reward


    def compute_termination_common(self, root_position):
        return torch.le(root_position[..., 2], self.termination_height)
