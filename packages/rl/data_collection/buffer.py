import torch
from utils import nest


class ReplayBuffer(object):
    """Replay Buffer."""

    def __init__(self, size):
        """Replay buffer for batched environments."""
        self.size = size
        self.num_envs = None  # determined lazily
        self.device = None    # determined lazily
        self.capacity = None  # determined lazily, equal to size * num_envs

        self.next_idx = 0
        self.num_in_buffer = 0

        self.data = None
        self.required_keys = ['obs', 'action', 'reward', 'done']

    def add_transition(self, transition):
        if self.data is None:
            self._init_replay_data(transition)

        def _insert(item):
            buf, x = item
            buf[self.next_idx].copy_(x)
        nest.map_structure(_insert, nest.zip_structure(self.data, transition))
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.num_in_buffer + self.num_envs, self.capacity)

    def sample(self, batch_size):
        inds = torch.randint(0, self.num_in_buffer - 1, size=(batch_size,), device=self.device)

        def _sample(buf, inds):
            return buf.view(buf.shape[0] * buf.shape[1], *buf.shape[2:])[inds]

        data = nest.map_structure(lambda buf: _sample(buf, inds), self.data)
        next_inds = (inds + 1) % self.capacity
        data['next_obs'] = nest.map_structure(lambda buf: _sample(buf, next_inds), self.data['obs'])
        return data

    def env_reset(self):
        if self.num_in_buffer > 0:
            self.data['done'][(self.next_idx-1) % self.size] = True

    def _init_replay_data(self, step_data):
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError("obs, action, reward, and done must be keys in the"
                                 "dict passed to buffer.store_effect.")
        self.num_envs = step_data['done'].shape[0]
        self.device = step_data['done'].device
        self.capacity = self.num_envs * self.size

        def _make_buffer(x):
            return torch.empty([self.size] + list(x.shape), dtype=x.dtype, device=x.device)
        self.data = nest.map_structure(_make_buffer, step_data)

    def state_dict(self):
        """State dict."""
        return {
            'data': self.data,
            'num_in_buffer': self.num_in_buffer,
            'next_idx': self.next_idx,
            'capacity': self.capacity,
            'num_envs': self.num_envs,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.num_in_buffer = state_dict['num_in_buffer']
        self.next_idx = state_dict['next_idx']
        self.num_envs = state_dict['num_envs']
        self.device = state_dict['device']
        self.capacity = state_dict['capacity']
        self.data = nest.map_structure(lambda buf: buf.to(self.device), state_dict['data'])


"""
Unit Tests
"""


if __name__ == '__main__':
    import unittest

    num_envs = 10

    def gen_transition():
        obs = [torch.rand((num_envs, 4, 5)), {'o1': torch.rand((num_envs, 3, 2)),
                                              'o2': torch.rand((num_envs, 3, 6))}]
        action = {'a1': torch.rand((num_envs, 3)), 'a2': torch.rand((num_envs, 1))}
        rew = torch.rand((num_envs,))
        done = torch.randint(0, 2, (num_envs,), dtype=torch.bool)
        return {'obs': obs, 'done': done, 'action': action, 'reward': rew,
                'other_data': torch.rand((num_envs, 5, 4))}

    class TestBuffer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            buffer = ReplayBuffer(10)
            for _ in range(10):
                buffer.add_transition(gen_transition())

            # Check sample shapes
            s = buffer.sample(2)
            assert s['obs'][0].shape == (2, 4, 5)
            assert s['obs'][1]['o1'].shape == (2, 3, 2)
            assert s['obs'][1]['o2'].shape == (2, 3, 6)
            assert s['next_obs'][0].shape == (2, 4, 5)
            assert s['next_obs'][1]['o1'].shape == (2, 3, 2)
            assert s['next_obs'][1]['o2'].shape == (2, 3, 6)
            assert s['other_data'].shape == (2, 5, 4)
            assert s['reward'].shape == (2,)
            assert s['done'].shape == (2,)
            assert s['action']['a1'].shape == (2, 3)
            assert s['action']['a2'].shape == (2, 1)

            # Check env reset
            buffer.env_reset()
            assert torch.all(buffer.data['done'][buffer.next_idx - 1 % buffer.size])

            # Check saving and loading
            state = buffer.state_dict()
            buffer2 = ReplayBuffer(10)
            buffer2.load_state_dict(state)
            assert torch.allclose(buffer.data['reward'], buffer2.data['reward'])


            for _ in range(10):
                t = gen_transition()
                buffer.add_transition(t)
                buffer2.add_transition(t)
            assert torch.allclose(buffer.data['reward'], buffer2.data['reward'])

    unittest.main()
