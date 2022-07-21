import os
import shutil
from typing import Optional
from collections import deque
import gym
import numpy as np
import torch
import wandb
from utils.mp_util import subproc_worker
from envs import IsaacMixedXMLEnv
from nlimb.designs import Design
from nlimb.design_distributions import DesignDistribution, GrammarDesignDist


class DesignLogger:
    """Records and keeps the history of designs and rewards."""

    def __init__(self):
        self.count = 0
        self.columns = ['count', 'design', 'reward']
        self.data = []
        wandb.define_metric("train/design_count", step_metric="train/step")
        wandb.define_metric("designs", step_metric="train/design_count")

    def log(self, design: Design, reward: float):
        if self.count % 1000 == 0:
            self.data.append([self.count, design.to_str(), reward])
        if self.count % 10000 == 0:
            table = wandb.Table(data=self.data, columns=self.columns)
            wandb.log({"designs": table, "train/design_count": self.count})
        self.count += 1

    def get_design_count(self):
        return self.count


@subproc_worker
class XMLGenerator():
    def __init__(self, design_class):
        self.design_class = design_class

    def gen_xml(self, design_str: str, path: str):
        design = self.design_class.from_str(design_str)
        design.to_xml(path)


class DesignManager(gym.Wrapper):
    """Environment wrapper to handle design sampling/logging.

    This wrapper:
         - samples and sets design parameters at fixed intervals.
         - logs the performance of each design.
    """

    def __init__(self, env: IsaacMixedXMLEnv, design_dist: DesignDistribution,
                 steps_per_design: int, history_len: int, num_xml_workers: int = 64):
        super().__init__(env)
        self.steps_per_design = steps_per_design
        self.design_history = deque([], maxlen=history_len)
        self.reward_history = deque([], maxlen=history_len)
        self.rewards = torch.zeros((self.env.num_envs), device=self.env.device)
        self.steps = 0
        self.dist = design_dist
        self.subprocs = [XMLGenerator(design_dist.design_class) for _ in range(num_xml_workers)]
        self.designs = None
        self.logger = DesignLogger()
        self.design_count = 0
        self._eval = False

    def create_xmls(self, mode=False):
        """Create xml files."""
        if os.path.exists(self.env.asset_root):
            shutil.rmtree(self.env.asset_root)
        os.makedirs(self.env.asset_root)
        if mode:
            if isinstance(self.dist, GrammarDesignDist):
                mode, self.design_trace = self.dist.mode()
                self.designs = [mode]
            else:
                self.designs, self.design_trace = self.dist.mode(), None
        else:
            if isinstance(self.dist, GrammarDesignDist):
                self.designs, self.design_trace = self.dist.sample(self.env.num_envs)
            else:
                self.designs, self.design_trace = self.dist.sample(self.env.num_envs), None
        jobs = []
        for i, design in enumerate(self.designs):
            path = os.path.join(self.env.asset_root, f'{i:06d}.xml')
            worker = self.subprocs[i % len(self.subprocs)]
            jobs.append(worker.gen_xml(design.to_str(), path))
        for j in jobs:
            j.join()

    def get_design_count(self):
        return self.design_count

    def get_designs_and_rewards(self, n: int):
        return (list(self.design_history)[-n:],
                torch.Tensor(list(self.reward_history)[-n:]).to(self.rewards.device))

    def init_scene(self, evaluate=False, mode=False, create_xmls=True):
        self._eval = evaluate
        if create_xmls:
            self.create_xmls(mode=mode)
        self.env.init_scene()
        self.env.update_designs(self.designs)
        self.rewards.zero_()
        self.steps = 0

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if self.designs is None:
            self.init_scene()
        return self.env.reset(env_ids)

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        if self._eval:
            return ob, r, done, info
        self.steps += 1
        self.rewards += r
        return ob, r, done, info

    def time_for_design_reset(self):
        return self.steps_per_design > 0 and self.steps >= self.steps_per_design

    def reset_designs(self):
        for design, reward in zip(self.designs, self.rewards):
            self.logger.log(design, float(reward))
            self.design_history.append(design)
            self.reward_history.append(reward)
            self.design_count += 1
        self.init_scene()
