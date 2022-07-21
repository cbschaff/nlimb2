"""Some simple networks."""
from rl.modules import Policy, QFunction, PolicyBase, ActorCriticBase, ContinuousQFunctionBase
from rl.modules import Categorical, DiagGaussian, TanhDiagGaussian
from rl.networks.base import FeedForwardNet
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import mup
import math


class FeedForwardPolicyBase(PolicyBase):
    def __init__(self, observation_space, action_space, units=(32, 32), activation_fn=nn.ReLU,
                 squash_actions=False, use_mup=False):
        self.activation_fn = activation_fn
        self.units = units
        self.squash_actions = squash_actions
        self.use_mup = use_mup
        PolicyBase.__init__(self, observation_space, action_space)

    def build(self):
        self.net = FeedForwardNet(self.observation_space.shape[-1], units=self.units,
                                  activation_fn=self.activation_fn, activate_last=True)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_actions:
                self.dist = TanhDiagGaussian(self.units[-1], self.action_space.shape[0],
                                             use_mup=self.use_mup)
            else:
                self.dist = DiagGaussian(self.units[-1], self.action_space.shape[0],
                                         use_mup=self.use_mup)
        else:
            self.dist = Categorical(self.units[-1], self.action_space.n, use_mup=self.use_mup)

    def forward(self, obs):
        return self.dist(self.net(obs))

    def mup_init(self):
        self.dist.mup_init()
        for name, param in self.named_parameters():
            if 'bias' not in name and 'dist' not in name:
                mup.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'dist' not in name:
                nn.init.constant_(param, 0.0)


class FeedForwardActorCriticBase(ActorCriticBase):
    def __init__(self, observation_space, action_space, units=(32, 32), activation_fn=nn.ReLU,
                 squash_actions=False, use_mup=False):
        self.activation_fn = activation_fn
        self.units = units
        self.squash_actions = squash_actions
        self.use_mup = use_mup
        PolicyBase.__init__(self, observation_space, action_space)

    def build(self):
        self.net = FeedForwardNet(self.observation_space.shape[-1], units=self.units,
                                     activation_fn=self.activation_fn, activate_last=True)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_actions:
                self.dist = TanhDiagGaussian(self.units[-1], self.action_space.shape[0],
                                             use_mup=self.use_mup)
            else:
                self.dist = DiagGaussian(self.units[-1], self.action_space.shape[0],
                                         use_mup=self.use_mup)
        else:
            self.dist = Categorical(self.units[-1], self.action_space.n, use_mup=self.use_mup)
        if self.use_mup:
            self.value = mup.MuReadout(self.units[-1], 1)
        else:
            self.value = nn.Linear(self.units[-1], 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.dist(x), self.value(x)

    def mup_init(self):
        self.dist.mup_init()
        for name, param in self.named_parameters():
            if 'bias' not in name and 'dist' not in name:
                mup.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'dist' not in name:
                nn.init.constant_(param, 0.0)


def feed_forward_base(env, units, activation_fn, squash_actions, use_mup, predict_value):
    if predict_value:
        return FeedForwardActorCriticBase(env.observation_space, env.action_space, units,
                                          activation_fn, squash_actions, use_mup)
    else:
        return FeedForwardPolicyBase(env.observation_space, env.action_space, units, activation_fn,
                                     squash_actions, use_mup)


class FeedForwardQFunctionBase(ContinuousQFunctionBase):
    def __init__(self, observation_space, action_space, units=(32, 32), activation_fn=F.relu):
        self.activation_fn = activation_fn
        self.units = units
        ContinuousQFunctionBase.__init__(self, observation_space, action_space)

    def build(self):
        n_in = self.observation_space.shape[-1] + self.action_space.shape[-1]
        self.net = FeedForwardNet(n_in, units=list(self.units) + [1],
                                  activation_fn=self.activation_fn, activate_last=False)

    def forward(self, obs, ac):
        return self.net(torch.cat([obs, ac], dim=1))


@gin.configurable(module='rl')
def mlp_policy(env, units=(32, 32), activation_fn=nn.ReLU, predict_value=False,
               squash_actions=False, use_mup=True):
    if use_mup:
        model = feed_forward_base(env, units, activation_fn, squash_actions, use_mup=True,
                                  predict_value=predict_value)
        base_units = [64] * len(units)
        delta_units = [2] * len(units)
        base_model = feed_forward_base(env, base_units, activation_fn, squash_actions,
                                       use_mup=True, predict_value=predict_value)
        delta_model = feed_forward_base(env, delta_units, activation_fn, squash_actions,
                                        use_mup=True, predict_value=predict_value)

        mup.set_base_shapes(model, base_model, delta=delta_model)
        model.mup_init()
        del base_model
        del delta_model
        return Policy(model)
    else:
        base = feed_forward_base(env, units, activation_fn, squash_actions, use_mup=False,
                                 predict_value=predict_value)
        return Policy(base)


@gin.configurable(module='rl')
def mlp_tanh_gaussian_policy(*args, **kwargs):
    return mlp_policy(*args, squash_actions=True, **kwargs)


@gin.configurable(module='rl')
def mlp_continuous_qfunction(env, units=(32, 32), activation_fn=nn.ReLU):
    return QFunction(FeedForwardQFunctionBase(env.observation_space, env.action_space,
                                              units, activation_fn))
