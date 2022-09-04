import copy
import warnings
from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data._groundtruth import ConstrainedGroundTruthData
from visgrid.envs.taxi import TaxiEnv
from visgrid.sensors import *

class TaxiData(ConstrainedGroundTruthData):
    """
    Dataset that generates all possible taxi & passenger positions,
    in-taxi flag settings, and goal locations

    Based on https://github.com/nmichlo/disent and https://github.com/camall3n/visgrid
    """

    name = 'taxi'

    factor_names = (
        'taxi_row',
        'taxi_col',
        'passenger_row',
        'passenger_col',
        'in_taxi',
        'goal_idx',
    )

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return 5, 5, 5, 5, 2, 4

    def __init__(self, rgb: bool = True, transform=None):
        self._rgb = rgb
        sensor = SensorChain([
            NoiseSensor(sigma=0.01),
            ClipSensor(0.0, 1.0),
        ])
        self.env = TaxiEnv(
            size=5,
            n_passengers=1,
            exploring_starts=True,
            terminate_on_goal=False,
            image_observations=True,
            sensor=sensor,
            dimensions=self.dimensions,
        )
        self.env.reset()

        super().__init__(transform=transform)

    def _is_valid_pos(self, pos):
        taxi_row, taxi_col, psgr_row, psgr_col, in_taxi, goal_idx = pos
        if not in_taxi:
            return True
        elif (taxi_row == psgr_row) and (taxi_col == psgr_col):
            return True
        return False

    def _get_observation(self, idx):
        state = self.idx_to_pos(idx)
        obs = self.env.get_observation(state)

        if (not self._rgb):
            obs = np.mean(obs, axis=-1, keepdims=True)

        return obs.astype(np.float32)

class TaxiData64x64(TaxiData):
    dimensions = TaxiEnv.dimensions_64x64

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return 64, 64, (3 if self._rgb else 1)

class TaxiData84x84(TaxiData):
    dimensions = TaxiEnv.dimensions_84x84

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return 84, 84, (3 if self._rgb else 1)

class TaxiOracleData(TaxiData):

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return 6, 1, 1

    def _get_observation(self, idx):
        state = self.idx_to_pos(idx)
        obs = np.asarray(state).reshape(self.img_shape).astype(np.float32)
        return obs

    def render(self, *args):
        raise NotImplementedError("TaxiOracleData does not use render function")

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
