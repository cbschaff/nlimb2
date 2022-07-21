"""Data Storage for Rollouts."""

from functools import partial
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from utils import nest, misc


# don't recurse into PackedSequences
nest.add_item_class(torch.nn.utils.rnn.PackedSequence)


class RolloutStorage(object):
    """Rollout Storage.

    This class stores data from rollouts with an environment.

    Data is provided by passing a dictionary to the 'insert(data)' method.
    The data dictionary must have the keys:
        'obs', 'action', 'reward', 'done', and 'vpred'
    Any amount of additional data can be provided.

    'reward', 'done', and 'vpred' are assumed to be a single torch tensor.
    All other data may be arbitrarily nested torch tensors.

    Once all rollout data has been stored, it can be batched and iterated over
    by calling the 'sampler(batch_size)' method.
    """

    def __init__(self, num_steps):
        """Init."""
        self.num_steps = num_steps
        self.required_keys = ['obs', 'action', 'reward', 'done', 'vpred']
        self.rollout_complete = False
        self.step = 0
        # determined lazily when buffers are created
        self.device = None
        self.num_envs = None
        self.data = None
        self.vtarg = None
        self.atarg = None
        self.return_ = None
        self.q_mc = None
        self.deltas = None
        self.not_done = None
        self.inds = None
        self.n = None
        self.flattened_data = None

    def init_data(self, step_data):
        """Initialize data storage."""
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError(f"Key {k} must be provided in step_data.")
        if step_data['reward'].shape != step_data['vpred'].shape:
            raise ValueError('reward and vpred must have the same shape!')
        if step_data['reward'].shape != step_data['done'].shape:
            raise ValueError('reward and done must have the same shape!')

        def _make_storage(arr):
            shape = [self.num_steps] + list(arr.shape)
            return torch.zeros(size=shape, dtype=arr.dtype, device=self.device)

        self.device = step_data['reward'].device
        self.data = nest.map_structure(_make_storage, step_data)
        self.vtarg = torch.zeros_like(self.data['vpred'])
        self.atarg = torch.zeros_like(self.data['vpred'])
        self.return_ = torch.zeros_like(self.data['vpred'])
        self.q_mc = torch.zeros_like(self.data['vpred'])
        self.deltas = torch.zeros_like(self.data['vpred'])
        self.not_done = torch.zeros_like(self.data['done'])
        self.num_envs = step_data['reward'].shape[0]
        self.reset()

        self.n = self.num_envs * self.num_steps
        def _view(x):
            return x.view((self.n, *x.shape[2:]))
        self.flattened_data = nest.map_structure(_view, self.data)
        self.flattened_data['vtarg'] = self.vtarg.view((self.n,))
        self.flattened_data['atarg'] = self.atarg.view((self.n,))
        self.flattened_data['q_mc'] = self.q_mc.view((self.n,))
        self.flattened_data['return'] = self.return_.view((self.n,))
        self.inds = torch.arange(0, self.n, device=self.device)

    def reset(self):
        self.step = 0
        self.rollout_complete = False

    def deinit(self):
        self.data = None

    def insert(self, step_data):
        """Insert new data into storage.

        Transfers to the correct device if needed.
        """
        if self.data is None:
            self.init_data(step_data)

        if self.rollout_complete:
            raise ValueError("Tried to insert data when the rollout is "
                             " complete. Call rollout.reset() to reset.")

        def _copy_data(item):
            storage, step_data = item
            storage[self.step].copy_(step_data)

        nest.map_structure(_copy_data, nest.zip_structure(self.data, step_data))

        self.step += 1
        self.rollout_complete = self.step == self.num_steps

    def compute_targets(self, next_vpred, gamma, lambda_=1.0, norm_advantages=False):
        """Compute advantage targets."""
        if not self.rollout_complete:
            raise ValueError("Rollout should be complete before computing "
                             "targets.")
        self.not_done[:] = torch.logical_not(self.data['done'])
        self.q_mc[:] = misc.discount(self.data['reward'], gamma, self.not_done,
                                     end_value=next_vpred)
        self.return_[:] = misc.discount(self.data['reward'], 1.0, self.not_done)

        self.deltas.copy_(self.data['reward'])
        self.deltas[:-1] += self.not_done[:-1] * gamma * self.data['vpred'][1:]
        self.deltas[-1] += self.not_done[-1] * gamma * next_vpred
        self.deltas -= self.data['vpred']

        self.atarg.copy_(misc.discount(self.deltas, gamma * lambda_, self.not_done))
        self.vtarg.copy_(self.atarg + self.data['vpred'])
        if norm_advantages:
            self.atarg -= self.atarg.mean()
            self.atarg /= self.atarg.std()

    def _feed_forward_generator(self, batch_size):
        if not self.rollout_complete:
            raise ValueError("Finish rollout before batching data.")

        torch.randperm(self.n, out=self.inds)

        def _batch_data(data, indices):
            return data[indices]

        for i in range(self.n // batch_size):
            inds = self.inds[i * batch_size:(i+1) * batch_size]
            yield nest.map_structure(partial(_batch_data, indices=inds), self.flattened_data)
        if self.n % batch_size != 0:
            inds = self.inds[(self.n // batch_size):]
            yield nest.map_structure(partial(_batch_data, indices=inds), self.flattened_data)

    def sampler(self, batch_size):
        """Iterate over rollout data."""
        return self._feed_forward_generator(batch_size)


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestRollout(unittest.TestCase):
        """Test."""

        def test(self):
            """Test feeed forward generator."""
            def _gen_data(np, x, dones):
                data = {}
                data['obs'] = x*torch.ones(size=(np, 1, 84, 84))
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.ones(size=(np,))
                data['done'] = torch.Tensor(dones).bool()
                data['vpred'] = x*torch.ones(size=(np,))
                data['logp'] = torch.zeros(size=(np,))
                return data
            num_envs = 3
            r = RolloutStorage(10)
            for i in range(10):
                r.insert(_gen_data(num_envs, i, [i >= 7, i >= 9, i >= 7]))
                if i < 9:
                    try:
                        r.compute_targets(gamma=0.99, lambda_=1.0,
                                          norm_advantages=True)
                        assert False
                    except Exception:
                        pass
            next_vpred = _gen_data(num_envs, 11, [False] * num_envs)['vpred']
            r.compute_targets(next_vpred, gamma=0.99, lambda_=1.0, norm_advantages=True)
            assert (np.allclose(r.atarg.mean(), 0., atol=1e-6)
                    and np.allclose(r.atarg.std(), 1., atol=1e-6))

            for batch in r.sampler(2):
                assert batch['obs'].shape == (2, 1, 84, 84)
                assert batch['atarg'].shape == (2,)
                assert batch['vtarg'].shape == (2,)
                assert batch['done'].shape == (2,)
                assert batch['reward'].shape == (2,)
                assert batch['return'].shape == (2,)
                assert batch['q_mc'].shape == (2,)
                print(batch['return'])
                print(batch['q_mc'])
                print(batch['atarg'])

    unittest.main()
