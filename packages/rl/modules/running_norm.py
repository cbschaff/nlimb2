"""Running normalization.

https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
"""
import gym
import torch
import torch.nn as nn
from utils.misc import unpack_gym_space
from utils import nest
from functools import partial

# optimization flags for pytorch JIT
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

@torch.jit.script
def _update(mean: torch.Tensor,
            var: torch.Tensor,
            count: torch.Tensor,
            batch_mean: torch.Tensor,
            batch_var: torch.Tensor,
            batch_count: int):
    """Update mean and var."""
    delta = batch_mean - mean
    new_count = (count + batch_count)
    new_mean = mean + delta * (batch_count / new_count)
    new_var = count * var + batch_count * batch_var
    new_var = new_var + (delta**2) * count * batch_count / new_count
    new_var = new_var / new_count
    return new_mean, new_var, new_count, torch.sqrt(new_var)


class RunningNorm(nn.Module):
    """Normalize with running estimates of mean and variance."""

    def __init__(self, shape, eps=1e-5):
        """Init."""
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(size=shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(size=shape, dtype=torch.float), requires_grad=False)
        self.count = nn.Parameter(torch.zeros(size=[1], dtype=torch.float), requires_grad=False)
        self.std = nn.Parameter(torch.ones(size=shape, dtype=torch.float), requires_grad=False)
        self.eps = eps

    def forward(self, data, unnorm=False):
        """Forward."""
        if unnorm:
            return data.float() * (self.std + self.eps) + self.mean
        else:
            return (data.float() - self.mean) / (self.std + self.eps)

    def update(self, batch_mean, batch_var, batch_count):
        """Update mean and var."""
        self.mean[:], self.var[:], self.count[:], self.std[:] = _update(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def reset(self):
        self.mean.zero_()
        self.var.zero_()
        self.std.zero_()
        self.count.zero_()


class NestedRunningNorm(nn.Module):
    """Running norm for arbitrary gym observation spaces."""
    def __init__(self, gym_space, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.spaces = unpack_gym_space(gym_space)
        self.make_modules()

    def make_modules(self):
        modules = []
        for space in nest.flatten(self.spaces):
            if isinstance(space, gym.spaces.Box):
                modules.append(RunningNorm(space.shape, self.eps))
            else:
                modules.append(None)

        self.running_norms = nn.ModuleList([m for m in modules if m is not None])
        self.packed_norms = nest.pack_sequence_as(modules, nest.get_structure(self.spaces))

    def _norm(self, data, unnorm=False):
        rn, x = data
        if rn is not None:
            x = rn(x, unnorm)
        return x

    def forward(self, data, unnorm=False):
        return nest.map_structure(partial(self._norm, unnorm=unnorm),
                                  nest.zip_structure(self.packed_norms, data))

    def _update(self, data):
        rn, batch_mean, batch_var, batch_count = data
        if rn is not None:
            rn.update(batch_mean, batch_var, batch_count)

    def update(self, batch_mean, batch_var, batch_count):
        return nest.map_structure(self._update, nest.zip_structure(self.packed_norms, batch_mean,
                                                                   batch_var, batch_count))

    def reset(self):
        for rn in self.running_norms:
            rn.reset()


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestRN(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            ron = RunningNorm([5, 4])

            ob = torch.ones([5, 4])
            assert torch.allclose(ob, ron(ob), atol=2e-5)  # eps is 1e-5

            def var(i):
                return torch.var(torch.arange(1, i+1).float(), unbiased=False)

            for i in range(1, 6):
                # obs arriving in batches of 2: ((2*i-1) * ob, 2*i * ob)
                batch_mean = (2*i - 0.5) * ob
                batch_var = 0.25 * ob
                batch_count = 2

                ron.update(batch_mean, batch_var, batch_count)
                assert torch.allclose(ron.mean, (i+0.5) * ob, atol=2e-5)
                assert torch.allclose(ron.var, var(2*i) * ob, atol=2e-5)
                assert ron.count == 2*i
                assert torch.allclose(
                    ron(ob[None]),
                    (ob[None] - (i+0.5) * ob[None]) / torch.sqrt(
                        var(2*i) * ob[None]), atol=2e-5)

            assert ron(ob[None]).shape == ob[None].shape

            assert not ron.mean.requires_grad
            assert not ron.var.requires_grad
            assert not ron.count.requires_grad

            assert len(ron.state_dict()) == 4

        def test_nested(self):
            space = gym.spaces.Dict({
                'ob1': gym.spaces.Discrete(5),
                'ob2': gym.spaces.Tuple((gym.spaces.Box(low=-20, high=5, shape=(5,),
                                                        dtype=np.float32),
                                         gym.spaces.Discrete(2)))
            })
            ron = NestedRunningNorm(space)
            obs = [space.sample() for _ in range(10)]
            obs = nest.map_structure(np.stack, nest.zip_structure(*obs))
            obs = nest.map_structure(torch.from_numpy, obs)
            ron(obs)
            mean = nest.map_structure(lambda x: x.float().mean(dim=0), obs)
            var = nest.map_structure(lambda x: x.float().var(dim=0), obs)
            count = nest.map_structure(lambda x: x.shape[0], obs)
            ron.update(mean, var, count)

    unittest.main()
