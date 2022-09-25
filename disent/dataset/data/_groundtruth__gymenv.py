from typing import Tuple, Optional
import warnings

import numpy as np
import gym
import torch

from disent.dataset.data._groundtruth import IteratedGroundTruthData

class GymEnvData(IteratedGroundTruthData):
    """
    Dataset that generates states/observations for a Gym environment
    (possibly wrapped), using the env's reset() functionality.

    This dataset works provided that:
    1. any wrappers correctly update their `observation_space`, and ...
    2. the info dictionary contains the key `state` which contains
       factor information
    """

    name = 'gymenv'

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._factor_sizes

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self._factor_names

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self.env.observation_space.shape

    def __init__(self,
                 env: gym.Env,
                 seed: Optional[int] = None,
                 transform: callable = None,
                 sample_mode: Optional[str] = 'observations'):
        assert isinstance(env, gym.Env), 'env must be a Gym environment'
        self.env = env
        self.set_seed(seed)
        ob, info = self.env.reset(seed=seed)
        valid_modes = ['states', 'observations', 'transitions']
        if sample_mode is None:
            sample_mode = 'observations'
        if sample_mode not in valid_modes:
            raise ValueError(
                f"sample_mode must be in {valid_modes} but sample_mode was {repr(sample_mode)}")
        if sample_mode == 'states':
            warnings.warn("Note: sample_mode 'states' is just an alias for 'observations'.")
            sample_mode = 'observations'
        self.sample_mode = sample_mode
        self._factor_sizes = self.env.unwrapped.factor_sizes
        self._factor_names = tuple(f'f_{i}' for i in range(len(self._factor_sizes)))
        assert len(info['state']) == len(self.factor_sizes)
        assert ob.shape == self.img_shape
        super().__init__(transform=transform)

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_mode == 'observations':
            x = self.get_ob_state_pair()[0]
            if self._transform is not None:
                x = self._transform(x)
        elif self.sample_mode == 'transitions':
            x = self.get_ob_transition_pair()[0]
        return x

    def get_worker_init_fn(self):
        return self._worker_init_fn

    @staticmethod
    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset # the dataset copy in this worker process
        offset = 1000 # maximum number of seeds ever expected per trial
        dataset.set_seed(dataset.seed + worker_id * offset)

    def set_seed(self, seed=None):
        if seed is None:
            maxint = np.iinfo(np.uint32).max
            self.seed = self.env.np_random.integers(maxint)
        else:
            self.seed = seed
        self.env.np_random = np.random.default_rng(self.seed)

    def sample_state(self):
        return self.get_ob_state_pair()[-1]

    def get_ob_state_pair(self):
        ob, info = self.env.reset()
        return ob, info['state']

    def get_ob_transition_pair(self):
        ob, info = self.env.reset()
        action = self.env.action_space.sample()
        next_ob, reward, terminal, truncated, next_info = self.env.step(action)
        state = info['state']
        next_state = next_info['state']
        ob_transition = {
            'ob': ob,
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'truncated': truncated,
            'next_ob': next_ob,
        }
        state_transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'truncated': truncated,
            'next_state': next_state,
        }
        return ob_transition, state_transition

    def get_batch(self, batch_size: int):
        return self.sample_batch_with_factors(num_samples=batch_size)

    def sample_batch_with_factors(self, num_samples: int):
        """Sample a batch of observations X and factors Y."""
        ob_samples = []
        factor_samples = []
        for _ in range(num_samples):
            ob, state = self.get_ob_state_pair()
            if self._transform is not None:
                ob = self._transform(ob)
            ob_samples.append(ob)
            factor_samples.append(state)

        batch = np.stack(ob_samples)
        factors = np.stack(factor_samples)
        return batch, factors

    def sample_factors(self, size=None, f_idxs=None) -> np.ndarray:
        raise NotImplementedError('sample_factors not supported for IteratedStateSpace')
        # # get factor sizes
        # if f_idxs is not None:
        #     raise NotImplementedError('f_idxs is not supported in iterated state spaces')
        # f_sizes = self._factor_sizes

        # # get sample size
        # if size is None:
        #     shape = (1, )
        # elif isinstance(size, int):
        #     shape = (size, )
        # else:
        #     shape = size
        # n_samples = np.prod(shape)
        # shape = shape + (len(f_sizes), )

        # _, factor_samples = self.sample_batch_with_factors(n_samples)

        # if size is None:
        #     factor_samples = factor_samples[0]
        # else:
        #     factor_samples = factor_samples.reshape(size, len(f_sizes))

        # return factor_samples

    def sample_missing_factors(self, *args, **kwargs):
        raise NotImplementedError('sample_missing_factors not supported for IteratedStateSpace')

    def resample_other_factors(self, *args, **kwargs):
        raise NotImplementedError('resample_other_factors not supported for IteratedStateSpace')

    def resample_given_factors(self, *args, **kwargs):
        raise NotImplementedError('resample_given_factors not supported for IteratedStateSpace')

    def sample_random_factor_traversal(self, *args, **kwargs):
        raise NotImplementedError(
            f'sample_random_factor_traversal not implemented for IteratedStateSpace')

    def sample_random_factor_traversal_grid(self, *args, **kwargs):
        raise NotImplementedError(
            f'sample_random_factor_traversal_grid not implemented for IteratedStateSpace')
