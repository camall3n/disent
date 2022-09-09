from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data._groundtruth import ConstrainedGroundTruthData

class GymEnvData(ConstrainedGroundTruthData):
    """
    Dataset that generates all possible state positions for a Gym environment
    (possibly wrapped), provided that:
    1. any wrappers correctly update their `observation_space`, and ...
    2. the inner env implements the following:

        properties:
        - factor_sizes: tuple

        methods:
        - set_state(pos: tuple) -> None
        - is_valid_pos(pos: tuple) -> bool
        - get_observation(pos: tuple) -> gym.ObsType
    """

    name = 'gymenv'

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._factor_sizes

    @property
    def factor_names(self) -> Tuple[int, ...]:
        return self._factor_names

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self.env.observation_space.shape

    def __init__(self, env, transform=None):
        self.env = env
        self.wrappers = reversed(self._get_wrapper_chain(env))
        self.env.reset()
        self._factor_sizes = self.env.unwrapped.factor_sizes
        self._factor_names = tuple(f'f_{i}' for i in range(len(self._factor_sizes)))
        super().__init__(transform=transform)

    def _is_valid_pos(self, pos):
        return self.env.unwrapped.is_valid_pos(pos)

    def _get_wrapper_chain(self, env):
        wrappers = []
        while True:
            if callable(getattr(env, 'observation', False)):
                wrappers.append(env)
            if not getattr(env, 'env', False):
                break
            env = env.env
        return wrappers

    def _get_observation(self, idx):
        state = self.idx_to_pos(idx)
        self.env.unwrapped.set_state(state)
        obs = self.env.get_observation(state)
        for wrapper in self.wrappers:
            obs = wrapper.observation(obs)
        return obs.astype(np.float32)
