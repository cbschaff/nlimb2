"""SAC algorithm.

https://arxiv.org/abs/1801.01290
"""
import os
import torch
import numpy as np
import wandb
import gin
from utils import nest, misc, Checkpointer
from rl import Algorithm, ReplayBuffer, ReplayBufferDataManager, rl_evaluate
from rl import EpisodeLogger, ObsNormWrapper, ActionNormWrapper, RewardScaleWrapper


def soft_target_update(target_net, net, tau):
    """Soft update totarget network."""
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_((1. - tau) * tp.data + tau * p.data)


class SACActor(object):
    """SAC actor."""

    def __init__(self, pi):
        """Init."""
        self.pi = pi

    def __call__(self, obs):
        """Act."""
        return {'action': self.pi(obs).action}


@gin.configurable(denylist=['logdir'], module='rl')
class SAC(Algorithm):
    """SAC algorithm."""

    def __init__(self,
                 logdir,
                 env,
                 policy_fn,
                 qf_fn,
                 optimizer=torch.optim.Adam,
                 buffer_size=10000,
                 learning_starts=1000,
                 update_period=1,
                 batches_per_update=1,
                 batch_size=256,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 gamma=0.99,
                 reward_scale=1.0,
                 target_update_period=1,
                 policy_update_period=1,
                 target_smoothing_coef=0.005,
                 alpha=0.2,
                 automatic_entropy_tuning=True,
                 target_entropy=None,
                 eval_num_episodes=1,
                 num_recording_envs=0,
                 log_period=1000):
        """Init."""
        Algorithm.__init__(self, logdir)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.env = EpisodeLogger(ObsNormWrapper(ActionNormWrapper(env),
                                                update_norm_on_step=False),
                                 namespace='original')
        if reward_scale != 1.0:
            self.env = EpisodeLogger(RewardScaleWrapper(self.env, reward_scale),
                                     namespace='scaled')
        self.device = self.env.device
        self.eval_num_episodes = eval_num_episodes
        self.num_recording_envs = num_recording_envs
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.batches_per_update = batches_per_update
        self.batch_size = batch_size
        if target_update_period < self.update_period:
            self.target_update_period = self.update_period
        else:
            self.target_update_period = target_update_period - (
                                target_update_period % self.update_period)
        if policy_update_period < self.update_period:
            self.policy_update_period = self.update_period
        else:
            self.policy_update_period = policy_update_period - (
                                policy_update_period % self.update_period)
        self.target_smoothing_coef = target_smoothing_coef
        self.log_period = log_period

        self.pi = policy_fn(self.env).to(self.device)
        self.qf1 = qf_fn(self.env).to(self.device)
        self.qf2 = qf_fn(self.env).to(self.device)
        self.target_qf1 = qf_fn(self.env).to(self.device)
        self.target_qf2 = qf_fn(self.env).to(self.device)

        self.opt_pi = optimizer(self.pi.parameters(), lr=policy_lr)
        self.opt_qf1 = optimizer(self.qf1.parameters(), lr=qf_lr)
        self.opt_qf2 = optimizer(self.qf2.parameters(), lr=qf_lr)

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.buffer = ReplayBuffer(size=self.buffer_size)
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    SACActor(self.pi),
                                                    self.learning_starts,
                                                    self.update_period)

        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                target_entropies = nest.map_structure(
                    lambda space: -np.prod(space.shape).item(),
                    misc.unpack_gym_space(self.env.action_space)
                )
                self.target_entropy = sum(nest.flatten(target_entropies))

            self.log_alpha = torch.tensor(np.log([self.alpha]),
                                          requires_grad=True,
                                          device=self.device,
                                          dtype=torch.float32)
            self.opt_alpha = optimizer([self.log_alpha], lr=policy_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.opt_alpha = None

        self.mse_loss = torch.nn.MSELoss()

        # wandb.watch((self.pi, self.qf1, self.qf2), log="gradients", log_freq=100)
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        self.t = 0
        self.steps = 0

    def loss(self, batch):
        """Loss function."""
        pi_out = self.pi(batch['obs'], reparameterization_trick=True)
        logp = pi_out.dist.log_prob(pi_out.action)
        q1 = self.qf1(batch['obs'], batch['action']).value
        q2 = self.qf2(batch['obs'], batch['action']).value

        # alpha loss
        if self.automatic_entropy_tuning:
            ent_error = logp + self.target_entropy
            alpha_loss = -(self.log_alpha * ent_error.detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
            alpha_loss = 0

        # qf loss
        with torch.no_grad():
            next_pi_out = self.pi(batch['next_obs'])
            next_ac_logp = next_pi_out.dist.log_prob(next_pi_out.action)
            q1_next = self.target_qf1(batch['next_obs'], next_pi_out.action).value
            q2_next = self.target_qf2(batch['next_obs'], next_pi_out.action).value
            qnext = torch.min(q1_next, q2_next) - alpha * next_ac_logp
            qtarg = batch['reward'] + torch.logical_not(batch['done']) * self.gamma * qnext

        assert qtarg.shape == q1.shape
        assert qtarg.shape == q2.shape
        qf1_loss = self.mse_loss(q1, qtarg)
        qf2_loss = self.mse_loss(q2, qtarg)

        # pi loss
        pi_loss = None
        if self.steps % self.policy_update_period == 0:
            q1_pi = self.qf1(batch['obs'], pi_out.action).value
            q2_pi = self.qf2(batch['obs'], pi_out.action).value
            min_q_pi = torch.min(q1_pi, q2_pi)
            assert min_q_pi.shape == logp.shape
            pi_loss = (alpha * logp - min_q_pi).mean()

            # log pi loss about as frequently as other losses
            if self.t % self.log_period < self.policy_update_period:
                log_data = {}
                log_data['train/loss/pi'] = pi_loss.detach().cpu().numpy()
                ent = -torch.mean(logp.detach()).cpu().numpy().item()
                log_data['train/alg/entropy'] = ent
                log_data['train/alg/perplexity'] = np.exp(ent)
                if self.automatic_entropy_tuning:
                    log_data['train/loss/log_alpha'] = self.log_alpha.detach().cpu().numpy()
                    log_data['train/alg/entropy_target'] = self.target_entropy
                log_data['train/loss/qf1'] = qf1_loss.detach().cpu().numpy()
                log_data['train/loss/qf2'] = qf2_loss.detach().cpu().numpy()
                log_data['train/alg/qf1'] = q1.mean().detach().cpu().numpy()
                log_data['train/alg/qf2'] = q2.mean().detach().cpu().numpy()
                log_data['train/step'] = self.t
                wandb.log(log_data)
        return pi_loss, qf1_loss, qf2_loss

    def step(self):
        """Step optimization."""
        if self.t == 0:
            self.env.update_obs_norm_with_random_transitions(1000)
            self.env.finalize_obs_norm()
        self.t += self.data_manager.step_until_update()
        self.steps += 1
        if self.steps % self.target_update_period == 0:
            soft_target_update(self.target_qf1, self.qf1,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf2, self.qf2,
                               self.target_smoothing_coef)

        if self.steps % self.update_period == 0:
            for _ in range(self.batches_per_update):
                batch = self.data_manager.sample(self.batch_size)

                pi_loss, qf1_loss, qf2_loss = self.loss(batch)

                # update
                if pi_loss:
                    self.opt_pi.zero_grad()
                    pi_loss.backward()
                    self.opt_pi.step()

                self.opt_qf1.zero_grad()
                qf1_loss.backward()
                self.opt_qf1.step()

                self.opt_qf2.zero_grad()
                qf2_loss.backward()
                self.opt_qf2.step()

        return self.t

    def evaluate(self):
        """Evaluate."""
        outdir = os.path.join(self.logdir, 'eval/{self.t:09d}')
        os.makedirs(outdir)
        rl_evaluate(self.env, self.pi, self.eval_num_episodes, self.t, outdir,
                    self.num_recording_envs, record_viewer=True, log_to_wandb=True)
        self.data_manager.manual_reset()

    def save(self):
        """Save."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf1': self.opt_qf1.state_dict(),
            'opt_qf2': self.opt_qf2.state_dict(),
            'log_alpha': (self.log_alpha if self.automatic_entropy_tuning
                          else None),
            'opt_alpha': (self.opt_alpha.state_dict()
                          if self.automatic_entropy_tuning else None),
            'env': misc.env_state_dict(self.env),
            't': self.t
        }
        self.ckptr.save(state_dict, self.t)

        # save buffer seperately and only once (because it can be huge)
        buffer_ckpt = os.path.join(self.logdir, 'buffer.pt')
        torch.save(self.buffer.state_dict(), buffer_ckpt)

    def load(self, t=None):
        """Load."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.pi.load_state_dict(state_dict['pi'])
        self.qf1.load_state_dict(state_dict['qf1'])
        self.qf2.load_state_dict(state_dict['qf2'])
        self.target_qf1.load_state_dict(state_dict['target_qf1'])
        self.target_qf2.load_state_dict(state_dict['target_qf2'])

        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_qf1.load_state_dict(state_dict['opt_qf1'])
        self.opt_qf2.load_state_dict(state_dict['opt_qf2'])

        if state_dict['log_alpha']:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict['log_alpha'])
            self.opt_alpha.load_state_dict(state_dict['opt_alpha'])
        misc.env_load_state_dict(self.env, state_dict['env'])
        self.t = state_dict['t']

        buffer_ckpt = os.path.join(self.logdir, 'buffer.pt')
        if os.path.exists(buffer_ckpt):
            buffer_state = torch.load(buffer_ckpt)
            self.buffer.load_state_dict(buffer_state)
            self.data_manager.manual_reset()
        return self.t

    def close(self):
        """Close environment."""
        try:
            self.env.close()
        except Exception:
            pass
