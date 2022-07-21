"""PPO RL algorithm.

https://arxiv.org/abs/1707.06347
"""
import os
import torch
import torch.nn as nn
import wandb
import gin
from utils import nest, misc, Checkpointer
from rl import Algorithm, RolloutDataManager, rl_evaluate, ValueNormWrapper
from rl import EpisodeLogger, ObsNormWrapper, RewardScaleWrapper, ActionNormWrapper, TimeoutWrapper
from rl import CoOptObsNormWrapper


class PPOActor(object):
    """Actor."""

    def __init__(self, pi):
        """Init."""
        self.pi = pi

    def __call__(self, ob, state_in=None):
        """Produce decision from model."""
        outs = self.pi(ob, state_in)
        data = {'action': outs.action,
                'value': outs.value,
                'logp': outs.dist.log_prob(outs.action),
                'dist': outs.dist.to_tensors()}
        if outs.state_out is not None:
            data['state'] = outs.state_out
        return data


class AdaptiveLRScheduler():
    def __init__(self, opt, target, initial_lr, lr_fac=1.5, max_lr=1e-2,
                 min_lr=1e-6):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_fac = lr_fac
        self.target = target
        self.opt = opt
        self.lr = initial_lr

    def update(self, x):
        if x > 2.0 * self.target:
            new_lr = max(self.lr / self.lr_fac, self.min_lr)
            self._update_lr(new_lr / self.lr)
            self.lr = new_lr
        elif x < 0.5 * self.target:
            new_lr = min(self.lr * self.lr_fac, self.max_lr)
            self._update_lr(new_lr / self.lr)
            self.lr = new_lr

    def _update_lr(self, fac):
        for param_group in self.opt.param_groups:
            param_group['lr'] *= fac

    def change_target(self, new_target):
        self.target = new_target

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']


class DecayScheduler():
    def __init__(self, period, inital_value, decay_factor):
        self.period = period
        self.value = inital_value
        self.fac = decay_factor
        self._best_metric = None
        self._t = 0

    def update_target(self, metric, t):
        if self._best_metric is None or metric > self._best_metric:
            self._best_metric = metric
            self._t = t

        if t - self._t >= self.period:
            self.value *= self.fac
            self._t = t
        return self.value

    def state_dict(self):
        return {
            't': self._t,
            'value': self.value,
            'metric': self._best_metric
        }

    def load_state_dict(self, state_dict):
        self._t = state_dict['t']
        self.value = state_dict['value']
        self._best_metric = state_dict['metric']


@gin.configurable(denylist=['logdir'], module='rl')
class PPO(Algorithm):
    """PPO algorithm."""

    def __init__(self,
                 logdir,
                 env,
                 policy_fn,
                 optimizer=torch.optim.Adam,
                 lr=1e-3,
                 batch_size=32,
                 batches_per_update=1,
                 rollout_length=128,
                 gamma=0.99,
                 lambda_=0.95,
                 epochs_per_rollout=10,
                 max_grad_norm=None,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 bounds_coef=0.001,
                 clip_param=0.2,
                 use_clipped_value_loss=False,
                 reward_scale=1.0,
                 kl_target=None,
                 kl_decay_period=None,
                 kl_decay_fac=0.5,
                 kl_lr_update_fac=1.5,
                 max_lr=0.01,
                 min_lr=1e-6,
                 norm_observations=True,
                 use_masked_obs_norm=False,
                 norm_values=False,
                 norm_advantages=False,
                 eval_num_episodes=1,
                 eval_max_episode_length=1000,
                 num_recording_envs=0,
                 record_viewer=True):
        """Init."""
        Algorithm.__init__(self, logdir)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        if norm_observations:
            if use_masked_obs_norm:
                self.env = CoOptObsNormWrapper(env, update_norm_on_step=True)
            else:
                self.env = ObsNormWrapper(ActionNormWrapper(env), update_norm_on_step=True)
        else:
            self.env = ActionNormWrapper(env)
        if reward_scale != 1.0:
            self.env = RewardScaleWrapper(self.env, reward_scale)
        self.env = EpisodeLogger(self.env, namespace='')

        self.lr = lr
        self.device = self.env.device
        self.eval_num_episodes = eval_num_episodes
        self.num_recording_envs = num_recording_envs
        self.record_viewer = record_viewer
        self.epochs_per_rollout = epochs_per_rollout
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.bounds_coef = bounds_coef
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.eval_max_episode_length = eval_max_episode_length
        self.norm_observations = norm_observations
        self.norm_values = norm_values
        self.batches_per_update = batches_per_update

        if norm_values:
            self.pi = ValueNormWrapper(policy_fn(self.env)).to(self.device)
            self.opt = optimizer(self.pi.module.parameters(), lr=lr)
        else:
            self.pi = policy_fn(self.env).to(self.device)
            self.opt = optimizer(self.pi.parameters(), lr=lr)

        self.data_manager = RolloutDataManager(
            self.env,
            PPOActor(self.pi),
            batch_size=batch_size,
            rollout_length=rollout_length,
            gamma=gamma,
            lambda_=lambda_,
            norm_advantages=norm_advantages)

        if kl_target is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = AdaptiveLRScheduler(self.opt, kl_target, lr, kl_lr_update_fac,
                    min_lr=min_lr, max_lr=max_lr)
            if kl_decay_period is None:
                self.kl_decay = None
            else:
                self.kl_decay = DecayScheduler(kl_decay_period, kl_target, kl_decay_fac)
        self.kl_target = kl_target

        self.mse = nn.MSELoss(reduction='none')

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("loss/*", step_metric="train/iterations")

        self.t = 0
        self.num_updates = 0
        self.iterations = 0

    def reset_rollout_storage(self):
        self.data_manager.deinit_storage()

    def compute_total_kl(self):
        """KL(old || new)."""
        kl = 0
        n = 0
        with torch.no_grad():
            for batch in self.data_manager.sampler():
                outs = self.pi(batch['obs'])
                approx_kl = self._compute_kl(batch['logp'], outs.dist.log_prob(batch['action']))
                s = nest.flatten(batch['action'])[0].shape[0]
                kl = (n / (n + s)) * kl + (s / (n + s)) * approx_kl
                n += s
        return kl

    def _compute_kl(self, old_logp, new_logp):
        # Calculate approximate form of reverse KL Divergence for early stopping
        # See Schulman blog: http://joschu.net/blog/kl-approx.html
        log_ratio = new_logp - old_logp
        return torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

    def compute_total_exact_kl(self):
        """KL(old || new)."""
        kl = 0
        n = 0
        with torch.no_grad():
            for batch in self.data_manager.sampler():
                outs = self.pi(batch['obs'])
                old_dist = outs.dist.from_tensors(batch['dist'])
                batch_kl = old_dist.kl(outs.dist).mean()
                s = nest.flatten(batch['action'])[0].shape[0]
                kl = (n / (n + s)) * kl + (s / (n + s)) * batch_kl
                n += s
        return kl

    def loss(self, batch):
        """Compute loss."""
        if self.norm_values:
            outs = self.pi(batch['obs'], unnorm_value=False)
        else:
            outs = self.pi(batch['obs'])
        loss = {}

        # compute policy loss
        logp = outs.dist.log_prob(batch['action'])
        assert logp.shape == batch['logp'].shape
        ratio = torch.exp(logp - batch['logp'])
        assert ratio.shape == batch['atarg'].shape
        ploss1 = ratio * batch['atarg']
        ploss2 = torch.clamp(ratio, 1.0-self.clip_param,
                             1.0+self.clip_param) * batch['atarg']
        pi_loss = -torch.min(ploss1, ploss2).mean()
        loss['pi'] = pi_loss

        # compute value loss
        assert outs.value.shape == batch['vtarg'].shape
        assert batch['vpred'].shape == batch['vtarg'].shape
        if self.use_clipped_value_loss:
            vloss1 = 0.5 * self.mse(outs.value, batch['vtarg'])
            vpred_clipped = batch['vpred'] + (
                outs.value - batch['vpred']).clamp(-self.clip_param,
                                                   self.clip_param)
            vloss2 = 0.5 * self.mse(vpred_clipped, batch['vtarg'])
            vf_loss = torch.max(vloss1, vloss2).mean()
        else:
            vf_loss = 0.5 * self.mse(outs.value, batch['vtarg']).mean()
        loss['value'] = vf_loss

        # compute entropy loss
        ent_loss = outs.dist.entropy().mean()
        loss['entropy'] = ent_loss

        # compute bounds loss
        # because of action norm wrapper, actions should be between -1, 1
        mu = outs.dist.mode()
        high = torch.clamp(mu - 1.1, min=0.0) ** 2
        low = torch.clamp(-mu - 1.1, min=0.0) ** 2
        bounds_loss = torch.mean(low + high)
        loss['bounds'] = bounds_loss

        tot_loss = (pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                    + self.bounds_coef * bounds_loss)
        loss['total'] = tot_loss
        return loss

    def step(self):
        """Compute rollout, loss, and update model."""
        self.pi.train()
        self.t += self.data_manager.rollout()
        self._all_log_data = None
        self._count = 0.
        # this is for logging, but has to be computed before values are normalized
        data = self.data_manager.storage.flattened_data
        value_error = data['vpred'] - data['q_mc']
        value_var = data['q_mc'].var()
        if self.norm_values:
            data = self.data_manager.storage.flattened_data
            self.pi.update_value_norm(data['vpred'])
            self.pi.update_value_norm(data['vtarg'])
            data['vpred'].copy_(torch.clamp(self.pi.value_norm(data['vpred']), -5, 5))
            data['vtarg'].copy_(torch.clamp(self.pi.value_norm(data['vtarg']), -5, 5))

        def _maybe_update(nbatches, loss):
            if nbatches % self.batches_per_update == 0:
                log_data = {
                    f'loss/{k}': v.detach().clone() for k, v in loss.items()
                }
                if self.max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
                    log_data['loss/grad_norm'] = norm
                self.opt.step()
                self.opt.zero_grad()

                if self._all_log_data is None:
                    self._all_log_data = log_data
                else:
                    with torch.no_grad():
                        self._all_log_data = nest.map_structure(sum,
                                                          nest.zip_structure(self._all_log_data,
                                                                             log_data))
                self.num_updates += 1
                self._count += 1

        for _ in range(self.epochs_per_rollout):
            self.opt.zero_grad()
            nbatches = 0
            for i, batch in enumerate(self.data_manager.sampler()):
                loss = self.loss(batch)
                loss['total'].backward()
                nbatches += 1
                _maybe_update(nbatches, loss)
            if nbatches % self.batches_per_update != 0:
                _maybe_update(0, loss)

        all_log_data = nest.map_structure(lambda x: x / self._count, self._all_log_data)

        kl = self.compute_total_kl()
        if self.lr_scheduler is not None:
            if self.kl_decay is not None:
                new_kl_target = self.kl_decay.update_target(self.env.best_reward, self.t)
                if new_kl_target != self.kl_target:
                    print("Decay KL!")
                    self.kl_target = new_kl_target
                    self.lr_scheduler.change_target(self.kl_target)
                    # for module in self.pi.modules():
                    #     if isinstance(module, nn.Dropout):
                    #         print('Decaying Dropout!')
                    #         module.p *= self.kl_decay.fac
            self.lr_scheduler.update(kl)

        # log once per iteration
        self.iterations += 1
        all_log_data['train/value_error_mean'] = value_error.mean()
        all_log_data['train/value_error_std'] = value_error.std()
        all_log_data['train/value_explained_variance'] = 1.0 - value_error.var() / value_var
        all_log_data['train/kl'] = kl
        all_log_data['train/kl_target'] = self.kl_target
        all_log_data['train/iterations'] = self.iterations
        all_log_data['train/step'] = self.t
        all_log_data['train/lr'] = self.lr if self.lr_scheduler is None else self.lr_scheduler.lr
        if self.norm_values:
            all_log_data['train/value_norm_mean'] = self.pi.value_norm.mean.cpu().numpy().item()
            all_log_data['train/value_norm_std'] = self.pi.value_norm.std.cpu().numpy().item()
        wandb.log(all_log_data)
        return self.t

    def evaluate(self):
        """Evaluate model."""
        outdir = os.path.join(self.logdir, f'eval/{self.t:012d}')
        os.makedirs(outdir)
        eval_env = TimeoutWrapper(self.env, self.eval_max_episode_length)
        rl_evaluate(eval_env, self.pi, self.eval_num_episodes, self.t, outdir,
                    self.num_recording_envs, record_viewer=self.record_viewer,
                    log_to_wandb=True)

    def save(self):
        """State dict."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'opt': self.opt.state_dict(),
            'env': misc.env_state_dict(self.env),
            't': self.t,
            'num_updates': self.num_updates,
            'iterations': self.iterations,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'kl_decay': self.kl_decay.state_dict() if self.kl_decay is not None else None
        }
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        """Load state dict."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.pi.load_state_dict(state_dict['pi'])
        self.opt.load_state_dict(state_dict['opt'])
        misc.env_load_state_dict(self.env, state_dict['env'])
        self.t = state_dict['t']
        self.num_updates = state_dict['num_updates']
        self.iterations = state_dict['iterations']
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        if self.kl_decay is not None:
            self.kl_decay.load_state_dict(state_dict['kl_decay'])
            self.lr_scheduler.change_target(self.kl_decay.value)
        return self.t

    def close(self):
        """Close environment."""
        try:
            self.env.close()
        except Exception:
            pass

    def manual_reset(self):
        self.data_manager.manual_reset()
