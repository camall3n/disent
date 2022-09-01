import copy
import warnings
from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data._groundtruth import ConstrainedGroundTruthData
from visgrid.taxi.taxi import BaseTaxi as TaxiEnv
from visgrid.gridworld.objects.passenger import Passenger
from visgrid.gridworld.objects.depot import Depot
from visgrid.gridworld.objects.agent import Agent as TaxiAgent

class TaxiData(ConstrainedGroundTruthData):
    """
    Dataset that generates all possible taxi & passenger positions,
    in-taxi flag settings, and goal locations

    Based on https://github.com/nmichlo/disent and https://github.com/camall3n/visgrid
    """

    name = 'taxi'

    factor_names = ('taxi_row', 'taxi_col', 'passenger_row', 'passenger_col', 'in_taxi',
                    'goal_idx')

    _depot_locs = {# yapf: disable
        'red':    (0, 0),
        'yellow': (4, 0),
        'blue':   (4, 3),
        'green':  (0, 4),
    }# yapf: enable
    _depot_names = list(_depot_locs.keys())

    _rows = 5
    _cols = 5

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return 5, 5, 5, 5, 2, 4

    def __init__(self, rgb: bool = True, transform=None):
        self._rgb = rgb
        self._grid = np.ones([self._rows * 2 + 1, self._cols * 2 + 1], dtype=int)
        # Reset valid positions and walls
        self._grid[1:-1:2, 1:-1] = 0
        self._grid[1:-1, 1:-1:2] = 0
        self._grid[1:4, 4] = 1
        self._grid[7:10, 2] = 1
        self._grid[7:10, 6] = 1

        # Place depots
        self.depots = dict()
        for name in self._depot_names:
            self.depots[name] = Depot(color=name)
            self.depots[name].position = self._depot_locs[name]

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
        taxi_row, taxi_col, psgr_row, psgr_col, in_taxi, goal_idx = state
        taxi = TaxiAgent(position=(taxi_row, taxi_col))
        passenger = Passenger(position=(psgr_row, psgr_col))
        passenger.color = self._depot_names[goal_idx]
        passenger.goal = passenger.color
        passenger.in_taxi = in_taxi

        img_height = (self._rows * self.cell_width + (self._rows + 1) * self.wall_width +
                      sum(self.banner_widths))
        img_width = (self._cols * self.cell_width + (self._cols + 1) * self.wall_width +
                     sum(self.banner_widths))

        dims = {
            'wall_width': self.wall_width,
            'cell_width': self.cell_width,
            'passenger_width': self.passenger_width,
            'depot_width': self.depot_width,
            'banner_widths': self.banner_widths,
            'dash_widths': self.dash_widths,
            'img_shape': (img_height, img_width)
        }

        obs = TaxiEnv._render(self._grid, taxi, [passenger], self.depots, dims)

        if (not self._rgb):
            obs = np.mean(obs, axis=-1, keepdims=True)

        return obs.astype(np.float32)

class TaxiData64x64(TaxiData):
    wall_width = 1
    cell_width = 11
    passenger_width = 7
    depot_width = 2
    banner_widths = (2, 1)
    dash_widths = (4, 4)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return 64, 64, (3 if self._rgb else 1)

class TaxiData84x84(TaxiData):
    wall_width = 2
    cell_width = 13
    passenger_width = 9
    depot_width = 3
    banner_widths = (4, 3)
    dash_widths = (6, 6)

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
