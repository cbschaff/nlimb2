"""Code for storing and iterating over rollout data."""
from rl.data_collection import RolloutStorage
from utils import nest
import torch


class RolloutDataManager(object):
    """Collects data from environments and stores it in a RolloutStorage.

    The resposibilities of this class are:
        - Handle storage of rollout data
        - Handle computing rollouts
        - Handle batching and iterating over rollout data

    act_fn:
        A callable which takes in the observation, recurrent state and returns:
            - a dictionary with the data to store in the rollout. 'action'
              and 'value' must be in the dict. Recurrent states must
              be nested under the 'state' key. All values except
              data['state'] must be pytorch Tensors.
    """

    def __init__(self,
                 env,
                 act_fn,
                 batch_size,
                 rollout_length,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False):
        """Init."""
        self.env = env
        self.nenv = self.env.num_envs
        self.act = act_fn
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_advantages = norm_advantages

        self.storage = RolloutStorage(self.rollout_length)
        self.init_state = None
        self._state = None
        self._ob = None
        self._not_done = torch.zeros((self.nenv,), dtype=torch.bool, device=self.env.device)
        self.recurrent = None

    def init_rollout_storage(self):
        """Initialize rollout storage."""
        self._ob = self.env.reset()
        data = self.act(self._ob)
        if 'action' not in data:
            raise ValueError('the key "action" must be in the dict returned '
                             'act_fn')
        if 'value' not in data:
            raise ValueError('the key "value" must be in the dict returned '
                             'act_fn')
        state = None
        if 'state' in data:
            state = data['state']

        if state is None:
            self.recurrent = False
        else:
            self.recurrent = True
            self.init_state = nest.map_structure(torch.zeros_like, state)
            self._state = self.init_state

    def _reset(self):
        if self._ob is None:
            self.init_rollout_storage()
        self.storage.reset()

    def manual_reset(self):
        self._ob = self.env.reset()

    def rollout_step(self):
        """Compute one environment step."""
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state)
            else:
                outs = self.act(self._ob, state_in=None)
        ob, r, done, _ = self.env.step(outs['action'])
        data = {}
        data['obs'] = self._ob
        data['action'] = outs['action']
        data['reward'] = r
        data['done'] = done
        data['vpred'] = outs['value']
        for key in outs:
            if key not in ['action', 'value', 'state']:
                data[key] = outs[key]

        self._ob = ob
        if self.recurrent:
            self._state = outs['state']
            self._state_reset(done)

        self.storage.insert(data)

    def _get_next_value(self):
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state)
            else:
                outs = self.act(self._ob, state_in=None)
        return outs['value']

    def _state_reset(self, dones):
        def _state_item_reset(x):
            x[0, dones].zero_()
        nest.map_structure(_state_item_reset, self._state)

    def rollout(self):
        """Compute entire rollout and advantage targets."""
        self._reset()
        while not self.storage.rollout_complete:
            self.rollout_step()
        self.storage.compute_targets(self._get_next_value(), self.gamma, self.lambda_,
                                     norm_advantages=self.norm_advantages)
        dones = self.env.get_data_quality_terminations()
        if dones is not None and torch.any(dones):
            self._ob = self.env.reset(dones.nonzero()[:, 0])
        return self.rollout_length * self.nenv

    def sampler(self):
        """Create sampler to iterate over rollout data."""
        return self.storage.sampler(self.batch_size)

    def deinit_storage(self):
        self.storage.deinit()


if __name__ == '__main__':
    import unittest
    from rl.modules import Policy, ActorCriticBase
    from rl.modules import Categorical, DiagGaussian
    from rl.networks import mlp_policy, FeedForwardNet
    import gym
    from torch.nn.utils.rnn import PackedSequence
    import numpy as np


    class DummyEnv():
        def __init__(self, num_envs, ob_space, ac_space):
            self.num_envs = num_envs
            self.observation_space = ob_space
            self.action_space = ac_space

        def _make_ob(self):
            obs = [self.observation_space.sample() for _ in range(self.num_envs)]
            return torch.from_numpy(np.stack(obs))

        def _sample_reward(self):
            return torch.rand((self.num_envs,))

        def _sample_dones(self):
            dones = [np.random.rand() < 0.1 for _ in range(self.num_envs)]
            return torch.from_numpy(np.array(dones))

        def sample_action(self):
            acs = [self.action_space.sample() for _ in range(self.num_envs)]
            return torch.from_numpy(np.stack(acs))

        def reset(self):
            return self._make_ob()

        def step(self, acs):
            return self._make_ob(), self._sample_reward(), self._sample_dones(), {}


    class RNNBase(ActorCriticBase):
        """Test recurrent network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32], activate_last=True)
            if hasattr(self.action_space, 'n'):
                self.dist = Categorical(32, self.action_space.n)
            else:
                self.dist = DiagGaussian(32, self.action_space.shape[0])
            self.lstm = torch.nn.LSTM(32, 32, 1)
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob, state_in=None):
            """Forward."""
            if isinstance(ob, PackedSequence):
                x = self.net(ob.data.float())
                x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                                   sorted_indices=ob.sorted_indices,
                                   unsorted_indices=ob.unsorted_indices)
            else:
                x = self.net(ob.float()).unsqueeze(0)
            if state_in is None:
                x, state_out = self.lstm(x)
            else:
                x, state_out = self.lstm(x, state_in['lstm'])
            if isinstance(x, PackedSequence):
                x = x.data
            else:
                x = x.squeeze(0)
            state_out = {'lstm': state_out, '1': torch.zeros_like(state_out[0])}
            return self.dist(x), self.vf(x), state_out

    def rnn_policy(env):
        return Policy(RNNBase(env.observation_space, env.action_space))

    class RolloutActor(object):
        """actor."""

        def __init__(self, pi):
            """init."""
            self.pi = pi

        def __call__(self, ob, state_in=None):
            """act."""
            outs = self.pi(ob, state_in)
            data = {'value': outs.value,
                    'action': outs.action}
            if outs.state_out:
                data['state'] = outs.state_out
            if isinstance(ob, (list, tuple)):
                data['key1'] = torch.zeros_like(ob[0])
            else:
                data['key1'] = torch.zeros_like(ob)
            return data

    def test(env, pi, batch_size, rollout_length):
        nenv = env.num_envs
        data_manager = RolloutDataManager(env, RolloutActor(pi),
                                          batch_size=batch_size,
                                          rollout_length=rollout_length)
        for _ in range(3):
            data_manager.rollout()
            count = 0
            for batch in data_manager.sampler():
                assert 'key1' in batch
                count += 1
                assert 'done' in batch
                data_manager.act(batch['obs'])
            if data_manager.recurrent:
                assert count == np.ceil(nenv / data_manager.batch_size)
            else:
                n = data_manager.storage.get_rollout()['reward'].data.shape[0]
                assert count == np.ceil(n / data_manager.batch_size)

    def env_discrete(nenv):
        """Create discrete env."""
        return DummyEnv(nenv, gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
                        gym.spaces.Discrete(4))

    def env_continuous(nenv):
        """Create continuous env."""
        return DummyEnv(nenv, gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
                        gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32))

    class TestRolloutDataCollection(unittest.TestCase):
        """Test case."""

        def test_feed_forward(self):
            """Test feed forward network."""
            env = env_discrete(2)
            test(env, mlp_policy(env, predict_value=True), 8, 128)

        def test_recurrent(self):
            """Test recurrent network."""
            env = env_discrete(2)
            test(env, rnn_policy(env), 2, 128)

        def test_feed_forward_continuous(self):
            """Test feed forward network."""
            env = env_continuous(2)
            test(env, mlp_policy(env, predict_value=True), 8, 128)

        def test_recurrent_continuous(self):
            """Test recurrent network."""
            env = env_continuous(2)
            test(env, rnn_policy(env), 2, 128)

    unittest.main()
