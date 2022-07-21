"""Joint optimization of design and control."""

from typing import Type, Optional
import inspect
import os
import shutil
import torch
import gin
import cv2
import wandb
import subprocess as sp
from rl import Algorithm, TimeoutWrapper
from utils import Checkpointer, subproc_worker
from envs import IsaacMixedXMLEnv
from utils.misc import ActionBounds
from nlimb.design_distributions import DesignDistribution, GrammarDesignDist
from .design_manager import DesignManager
from .design_opt import DesignDistOptimizer, GrammarDesignDistOptimizer
from rl.util import rl_evaluate
from utils import gin_util


# Due to a memory leak in IsaacGym, the environment must be run in a subprocess.
def subproc_env(env_cls):
    config = gin_util.get_config_dict()

    class GinWrapper(env_cls):
        def __init__(self, *args, **kwargs):
            import nlimb
            import tasks
            import envs
            import utils
            import rl
            gin_util.add_pytorch_external_configurables()
            gin_util.apply_bindings_from_dict(config)
            env_cls.__init__(self, *args, **kwargs)

    subproc_cls = subproc_worker(GinWrapper, ctx='spawn', daemon=True)

    class SubprocIsaacEnv():
        def __init__(self, *args, **kwargs):
            self.env = None
            self._args = args
            self._kwargs = kwargs
            self._create_env()
            self.num_envs = self.env.get_num_envs().results
            self.device = self.env.get_device().results
            self.observation_space = self.env.get_observation_space().results
            self.action_space = self.env.get_action_space().results
            self.asset_root = self.env.get_asset_root().results
            self.action_bounds = ActionBounds(self.action_space, self.device)

        def _create_env(self):
            if self.env is not None:
                self.env.close()
            self.env = subproc_cls(*self._args, **self._kwargs)

        def init_scene(self):
            self._create_env()

        def close(self):
            if self.env is not None:
                self.env.close()
                self.env = None

    def _add_command(name):
        def remote_fn(self, *args, **kwargs):
            return getattr(self.env, name)(*args, **kwargs).results
        setattr(SubprocIsaacEnv, name, remote_fn)

    for name, _ in inspect.getmembers(subproc_cls, inspect.isfunction):
        if name[0] == '_' or name in ['init_scene', 'close']:
            continue
        _add_command(name)

    return SubprocIsaacEnv


@gin.configurable(module='nlimb')
class NLIMB(Algorithm):
    def __init__(self,
                 logdir,
                 env: Type[IsaacMixedXMLEnv],
                 rl_algorithm: Type[Algorithm],
                 design_dist: Type[DesignDistribution], # The design distribution to be optimized
                 steps_per_design: int,  # Number of evironment steps each design takes
                 learning_starts: int,   # The number of designs to collect data with before learning
                 update_period: int,     # number of designs sampled between updates
                 n_updates: int,         # The number of updates per iteration
                 optimizer: Type[torch.optim.Optimizer],
                 lr: float,
                 n_epochs: int,
                 batch_size: int,
                 clip_param: float,
                 kl_target: float,
                 ent_coef: float,
                 lr_fac: float = 1.25,
                 max_lr: float = 0.1,
                 max_grad_norm: Optional[float] = None,
                 xml_root: str = '/xmls',
                 ):

        Algorithm.__init__(self, logdir)
        self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts_design_params'))
        self.design_dist = design_dist()
        self._init_xmls(xml_root)
        env = subproc_env(env)(asset_root=xml_root)
        self.design_dist.to(env.device)
        self.env = DesignManager(env, self.design_dist, steps_per_design,
                                 history_len=env.num_envs)
        self.alg = rl_algorithm(logdir, env=self.env)
        if isinstance(self.design_dist, GrammarDesignDist):
            design_opt = GrammarDesignDistOptimizer
        else:
            design_opt = DesignDistOptimizer
        self.design_opt = design_opt(self.design_dist, optimizer,
                                     lr=lr,
                                     n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     clip_param=clip_param,
                                     kl_target=kl_target,
                                     ent_coef=ent_coef,
                                     device=env.device,
                                     lr_fac=lr_fac,
                                     max_lr=max_lr,
                                     max_grad_norm=max_grad_norm)

        self.steps_per_design = steps_per_design
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.n_updates = n_updates
        self.t = 0
        self.last_design_update = 0
        wandb.define_metric('nlimb/*', step_metric='train/step')

    def _init_xmls(self, xml_root):
        if os.path.exists(xml_root):
            shutil.rmtree(xml_root)
        os.makedirs(xml_root)
        if isinstance(self.design_dist, GrammarDesignDist):
            self.design_dist.design_class.from_random().to_xml(os.path.join(xml_root, 'init.xml'))
        else:
            self.design_dist.sample(1)[0].to_xml(os.path.join(xml_root, 'init.xml'))

    def step(self):
        self.t = self.alg.step()

        if self.env.time_for_design_reset():
            self.env.reset_designs()
            self.alg.manual_reset()
            self.alg.reset_rollout_storage()

        design_count = self.env.get_design_count()
        designs_since_update = design_count - self.last_design_update

        if (design_count >= self.learning_starts
           and designs_since_update >= self.update_period):
            self.update()
            self.last_design_update = design_count
        return self.t

    def update(self):
        self.save()
        for _ in range(self.n_updates):
            self.env.init_scene(evaluate=True)
            eval_env = self.alg.env
            if self.alg.eval_max_episode_length is not None:
                eval_env = TimeoutWrapper(eval_env, self.alg.eval_max_episode_length)
            data = rl_evaluate(eval_env, self.alg.pi, self.env.num_envs, self.t, outdir=None,
                               record_viewer=False, log_to_wandb=False)
            rewards = torch.tensor(data['episode_rewards'], device=self.env.device)
            # designs, rewards = self.env.get_designs_and_rewards(self.env.num_envs)
            if isinstance(self.design_opt, GrammarDesignDistOptimizer):
                self.design_opt.update(self.env.design_trace, rewards, self.t)
            else:
                self.design_opt.update(self.env.designs, rewards, self.t)
            log_data = {'nlimb/eval_length': wandb.Histogram(data['episode_lengths']),
                        'nlimb/eval_reward': wandb.Histogram(data['episode_rewards']),
                        'train/step': self.t}
            log_data.update(self.design_dist.get_log_dict())
            wandb.log(log_data)
        self.env.init_scene()
        self.alg.manual_reset()


    def evaluate(self):
        self.env.init_scene(evaluate=True)
        self.alg.evaluate()
        self.env.init_scene()

    def save(self):
        self.alg.save()
        state_dict = {'opt': self.design_opt.state_dict(),
                      'dist': self.design_dist.state_dict(),
                      'design_count': self.env.get_design_count()}
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        self.t = self.alg.load(t)
        state_dict = self.ckptr.load(t)
        if state_dict is not None:
            self.design_opt.load_state_dict(state_dict['opt'])
            self.design_dist.load_state_dict(state_dict['dist'])
            self.env.design_count = state_dict['design_count']
        design_count = self.env.get_design_count()
        self.last_design_update = design_count - (design_count % self.update_period)
        return self.t

    def close(self):
        self.env.close()
        self.alg.close()
