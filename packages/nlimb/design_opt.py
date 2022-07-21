"""PPO-style optimizer for zero-order optimization."""

from typing import Type, Sequence, Optional
from functools import partial
from rl import RunningNorm, nest
from rl.algorithms.ppo import AdaptiveLRScheduler
from nlimb.designs import Design
from nlimb.design_distributions import DesignDistribution
import torch
import torch.nn as nn
import wandb


class EpochRunner():
    def __init__(self, data: dict, batch_size: int, n: int, device: str):
        self.data = data
        self.batch_size = batch_size
        self.n = n
        self.device = device
        self.inds = torch.arange(0, self.n, device=self.device)

    def epoch(self):
        torch.randperm(self.n, out=self.inds)

        def _batch_data(data, indices):
            if isinstance(data, torch.Tensor):
                return data[indices]
            elif hasattr(data, 'batch'):
                return data.batch(indices)
            else:
                return nest.NestTupleItem([data[ind] for ind in indices])

        for i in range(self.n // self.batch_size):
            inds = self.inds[i * self.batch_size:(i+1) * self.batch_size]
            yield nest.map_structure(partial(_batch_data, indices=inds), self.data)
        if self.n % self.batch_size != 0:
            inds = self.inds[(self.n // self.batch_size):]
            yield nest.map_structure(partial(_batch_data, indices=inds), self.data)


class DesignDistOptimizer():
    def __init__(self,
                 design_dist: Type[DesignDistribution], # The design distribution to be optimized
                 optimizer: Type[torch.optim.Optimizer],
                 lr: float,
                 n_epochs: int,
                 batch_size: int,
                 clip_param: float,
                 kl_target: float,
                 device: str,
                 lr_fac: float = 1.1,
                 max_lr: float = 0.1,
                 ent_coef: float = 0.0,
                 max_grad_norm: Optional[float] = None,
                 ):

        self.device = device
        self.design_dist = design_dist
        self.design_dist.to(device)
        self.opt = optimizer(self.design_dist.parameters(), lr=lr)
        self.reward_norm = RunningNorm((1,)).to(self.device)
        self.clip_param = clip_param
        self.kl_target = kl_target
        self.ent_coef = ent_coef
        self.lr_scheduler = AdaptiveLRScheduler(self.opt, kl_target, lr, lr_fac=lr_fac,
                                                max_lr=max_lr)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.t = 0
        wandb.define_metric('design_opt/*', step_metric='train/step')

    def update(self, designs: Sequence[Design], rewards: torch.Tensor, t: int):
        mean_reward = rewards.mean()
        self.reward_norm.update(mean_reward, rewards.var(), rewards.shape[0])
        normed_rewards = (rewards - mean_reward) / (self.reward_norm.std + 1e-5)
        with torch.no_grad():
            orig_logps = self.design_dist.log_prob(designs)
        data = {'designs': nest.NestTupleItem(designs), 'rewards': normed_rewards,
                'orig_logps': orig_logps}
        batcher = EpochRunner(data, self.batch_size, len(designs), self.device)

        all_log_data = None
        count = 0
        for _ in range(self.n_epochs):
            for batch in batcher.epoch():
                # PPO style update
                self.opt.zero_grad()
                logps = self.design_dist.log_prob(batch['designs'])
                ratio = torch.exp(logps - batch['orig_logps'])
                assert ratio.shape == batch['rewards'].shape
                loss1 = ratio * batch['rewards']
                loss2 = torch.clamp(ratio, 1.0-self.clip_param,
                                     1.0+self.clip_param) * batch['rewards']
                loss = -torch.min(loss1, loss2).mean()
                if self.ent_coef > 0.0:
                    entropy = self.design_dist.entropy(batch['designs'])
                    loss = loss - self.ent_coef * entropy

                loss.backward()
                log_data = {'design_opt/loss': loss.detach().cpu()}
                if self.max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(self.design_dist.parameters(),
                                                    self.max_grad_norm)
                    log_data['design_opt/grad_norm'] = norm
                    log_data['design_opt/grad_norm_clipped'] = min(norm, self.max_grad_norm)
                self.opt.step()
                if all_log_data is None:
                    all_log_data = log_data
                else:
                    all_log_data = nest.map_structure(sum, nest.zip_structure(all_log_data,
                                                                              log_data))
                count += 1

        # Calculate approximate form of reverse KL Divergence for early stopping
        # See Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            logps = self.design_dist.log_prob(designs)
            log_ratio = logps - orig_logps
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            entropy = self.design_dist.entropy(designs).cpu().numpy()

        self.lr_scheduler.update(approx_kl)
        all_log_data = nest.map_structure(lambda x: x / count, all_log_data)
        all_log_data['design_opt/kl'] = approx_kl
        all_log_data['design_opt/lr'] = self.lr_scheduler.lr
        all_log_data['design_opt/entropy'] = entropy
        all_log_data['train/step'] = t
        wandb.log(all_log_data)

    def state_dict(self):
        return {
            'opt': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict['opt'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])



class GrammarDesignDistOptimizer(DesignDistOptimizer):

    def update(self, trace, rewards: torch.Tensor, t: int):
        mean_reward = rewards.mean()
        self.reward_norm.update(mean_reward, rewards.var(), rewards.shape[0])
        normed_rewards = (rewards - mean_reward) / (self.reward_norm.std + 1e-5)
        with torch.no_grad():
            orig_logps, mask = self.design_dist.individual_log_prob(trace)
            normed_rewards = normed_rewards.unsqueeze(1).expand(-1, orig_logps.shape[1])
        data = {'designs': trace, 'rewards': normed_rewards,
                'orig_logps': orig_logps}
        batcher = EpochRunner(data, self.batch_size, len(rewards), self.device)
        all_log_data = None
        count = 0
        for _ in range(self.n_epochs):
            for batch in batcher.epoch():
                # PPO style update
                self.opt.zero_grad()
                logps, mask, ent = self.design_dist.individual_log_prob_and_ent(batch['designs'])
                assert mask.shape == logps.shape
                assert mask.shape == batch['orig_logps'].shape
                ratio = mask * torch.exp(logps - batch['orig_logps'])
                assert ratio.shape == batch['rewards'].shape
                loss1 = ratio * batch['rewards']
                loss2 = torch.clamp(ratio, 1.0-self.clip_param,
                                     1.0+self.clip_param) * batch['rewards']
                loss = -torch.min(loss1, loss2).sum() / torch.count_nonzero(mask)
                if self.ent_coef > 0.0:
                    loss = loss - self.ent_coef * ent
                loss.backward()
                log_data = {'design_opt/loss': loss.detach().cpu()}
                if self.max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(self.design_dist.parameters(),
                                                    self.max_grad_norm)
                    log_data['design_opt/grad_norm'] = norm
                    log_data['design_opt/grad_norm_clipped'] = min(norm, self.max_grad_norm)
                self.opt.step()
                if all_log_data is None:
                    all_log_data = log_data
                else:
                    all_log_data = nest.map_structure(sum, nest.zip_structure(all_log_data,
                                                                              log_data))
                count += 1

        # Calculate approximate form of reverse KL Divergence for early stopping
        # See Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            logps, mask = self.design_dist.individual_log_prob(trace)
            log_ratio = logps - orig_logps
            approx_kl = (torch.sum(mask * ((torch.exp(log_ratio) - 1) - log_ratio))
                         / torch.count_nonzero(mask)).cpu().numpy()
            entropy = self.design_dist.entropy(trace).cpu().numpy()

        self.lr_scheduler.update(approx_kl)
        all_log_data = nest.map_structure(lambda x: x / count, all_log_data)
        all_log_data['design_opt/kl'] = approx_kl
        all_log_data['design_opt/lr'] = self.lr_scheduler.lr
        all_log_data['design_opt/entropy'] = entropy
        all_log_data['train/step'] = t
        wandb.log(all_log_data)
