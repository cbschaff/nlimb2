"""Environment wrappers."""
from gym import ActionWrapper, spaces
import torch
import numpy as np
from utils.misc import ActionBounds
from utils import nest


class ActionNormWrapper(ActionWrapper):
    """Normalize the range of continuous action spaces."""

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        self.action_space = self.make_action_space(self.env.action_space)

    def make_action_space(self, ac_space):
        if isinstance(ac_space, spaces.Box):
            return spaces.Box(-np.ones_like(ac_space.low),
                              np.ones_like(ac_space.high), dtype=ac_space.dtype)
        elif isinstance(ac_space, spaces.Tuple):
            return spaces.Tuple([
                self.make_action_space(a_s) for a_s in ac_space
            ])
        elif isinstance(ac_space, spaces.Dict):
            return spaces.Dict({
                k: self.make_action_space(ac_space[k]) for k in ac_space.spaces
            })
        else:
            return ac_space

    def action(self, action):
        return self.env.action_bounds.unnorm(action)

    def reverse_action(self, action):
        return self.env.action_bounds.norm(action)

    def reset(self, env_ids=None):
        return self.env.reset(env_ids)


if __name__ == '__main__':

    import unittest
    import gym

    class StackActionWrapper(ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Tuple([
                spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                spaces.Dict({
                    'ac1': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                    'ac2': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
                })
            ])
            self.device = 'cpu'
            self.action_bounds = ActionBounds(self.action_space, self.device)

        def action(self, action):
            return action[0]

    class TestActionNormWrapper(unittest.TestCase):
        """Test DummyVecEnv"""

        def test(self):
            env = ActionNormWrapper(StackActionWrapper(gym.make('CartPole-v1')))

            env.reset()
            action = nest.map_structure(torch.from_numpy, env.action_space.sample())
            print(action)
            print(env.action(action))
            print(env.action_space)
            print(env.env.action_space)
            print(env.env.env.action_space)
            print(env.env.env.action_space)

    unittest.main()
