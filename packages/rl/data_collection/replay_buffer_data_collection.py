"""Collect data from an environment and store it in a replay buffer."""
import torch


class ReplayBufferDataManager(object):
    """Collects data from environments and stores it in a ReplayBuffer.

    act_fn:
        A callable which takes in the observation and returns:
            - a dictionary with the data to store in the replay buffer.
              'action' must be in the dict.
    """

    def __init__(self,
                 buffer,
                 env,
                 act_fn,
                 learning_starts=1000,
                 update_period=1):
        """Init."""
        self.env = env
        self.buffer = buffer
        self.act = act_fn
        self.learning_starts = learning_starts
        self.update_period = update_period
        self._ob = None

    def manual_reset(self):
        """Update buffer on manual environment reset."""
        self.buffer.env_reset()
        self._ob = self.env.reset()

    def env_step_and_store_transition(self):
        """Step env and store transition in replay buffer."""
        if self._ob is None:
            self.manual_reset()

        with torch.no_grad():
            data = self.act(self._ob)
        data['obs'] = self._ob
        self._ob, rew, done, _ = self.env.step(data['action'])
        data['reward'] = rew
        data['done'] = done
        self.buffer.add_transition(data)

    def step_until_update(self):
        """Step env untiil update."""
        t = 0
        for _ in range(self.update_period):
            self.env_step_and_store_transition()
            t += self.env.num_envs
        while self.buffer.num_in_buffer < min(self.learning_starts,
                                              self.buffer.capacity):
            self.env_step_and_store_transition()
            t += self.env.num_envs
        return t

    def sample(self, *args, **kwargs):
        """Sample batch from replay buffer."""
        return self.buffer.sample(*args, **kwargs)
