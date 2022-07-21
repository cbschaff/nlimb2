"""Value function module."""
import torch.nn as nn
import torch
from rl.modules.base import ValueFunctionBase
from collections import namedtuple
from rl.modules.running_norm import RunningNorm


class ValueFunction(nn.Module):
    """Value function module."""

    def __init__(self, base):
        """Init."""
        super().__init__()
        self.base = base
        assert isinstance(self.base, ValueFunctionBase)
        self.outputs = namedtuple('Outputs', ['value', 'state_out'])

    def forward(self, ob, state_in=None):
        """Forward."""
        outs = self.base(ob) if state_in is None else self.base(ob, state_in)
        if isinstance(outs, tuple):
            value, state_out = outs
        else:
            value, state_out = outs, None
        return self.outputs(value=value.squeeze(-1), state_out=state_out)


class ValueNormWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.value_norm = RunningNorm((1,))

    def forward(self, *args, unnorm_value=True, **kwargs):
        outs = self.module(*args, **kwargs)
        if unnorm_value:
            value = self.value_norm(outs.value, unnorm=True)
            new_outs = outs._asdict()
            new_outs['value'] = value
            return self.module.outputs(**new_outs)
        else:
            return outs

    def update_value_norm(self, values):
        mean = values.mean(dim=0)
        var = values.var(dim=0)
        count = values.shape[0]
        self.value_norm.update(mean, var, count)


if __name__ == '__main__':
    import gym
    import torch
    import unittest
    import numpy as np

    class TestVF(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            class Base(ValueFunctionBase):
                def forward(self, ob):
                    return torch.from_numpy(np.zeros((ob.shape[0], 1)))

            env = gym.make('CartPole-v1')
            vf = ValueFunction(Base(env.observation_space, env.action_space))
            ob = env.reset()

            assert np.allclose(vf(ob[None]).value, 0)

    unittest.main()
