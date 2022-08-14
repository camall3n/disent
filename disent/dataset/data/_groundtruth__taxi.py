import copy
import warnings
from typing import Optional
from typing import Tuple

import numpy as np
import matplotlib.colors as colors

from disent.dataset.data._groundtruth import GroundTruthData


class Depot:
    def __init__(self, position=(0, 0), color='red'):
        self.position = np.asarray(position)
        self.color = color

    def __setattr__(self, name, value):
        if name == 'position':
            value = copy.deepcopy(np.asarray(value, dtype=int))
        super().__setattr__(name, value)


class TaxiData(GroundTruthData):
    """
    Dataset that generates all possible taxi & passenger positions,
    in-taxi flag settings, and goal locations

    Based on https://github.com/nmichlo/disent and https://github.com/camall3n/visgrid
    """

    name = 'taxi'

    factor_names = ('taxi_row', 'taxi_col', 'passenger_row', 'passenger_col', 'in_taxi', 'goal_color')

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

    def _get_observation(self, idx):
        state = self.idx_to_pos(idx)
        obs = self.render(*state)
        return obs

    def render(self, taxi_row, taxi_col, psgr_row, psgr_col, in_taxi, goal_idx):
        wall_width =  self.wall_width
        cell_width = self.cell_width
        passenger_width = self.passenger_width
        depot_width =  self.depot_width
        banner_widths =  self.banner_widths
        dash_widths =  self.dash_widths
        img_width = self._cols * cell_width + (self._cols + 1) * wall_width + sum(banner_widths)
        img_height = self._rows * cell_width + (self._rows + 1) * wall_width + sum(banner_widths)
        img_shape = (img_height, img_width)

        walls = expand_grid(self._grid, cell_width, wall_width)
        walls = to_rgb(walls) * get_rgb('dimgray') / 8

        goal_color = self.depots[self._depot_names[goal_idx]].color

        passengers = np.zeros_like(walls)
        patch, marks = passenger_patch(cell_width, passenger_width, in_taxi)
        color_patch = to_rgb(patch) * get_rgb(goal_color)
        color_patch[marks > 0, :] = get_rgb('dimgray') / 4
        row, col = cell_start((psgr_row, psgr_col), cell_width, wall_width)
        passengers[row:row + cell_width, col:col + cell_width, :] = color_patch

        depots = np.zeros_like(walls)
        for depot in self.depots.values():
            patch = depot_patch(cell_width, depot_width)
            color_patch = to_rgb(patch) * get_rgb(depot.color)
            row, col = cell_start(depot.position, cell_width, wall_width)
            depots[row:row + cell_width, col:col + cell_width, :] = color_patch

        taxis = np.zeros_like(walls)
        patch = taxi_patch(cell_width, depot_width, passenger_width)
        color_patch = to_rgb(patch) * get_rgb('dimgray') / 4
        row, col = cell_start((taxi_row, taxi_col), cell_width, wall_width)
        taxis[row:row + cell_width, col:col + cell_width, :] = color_patch

        # compute foreground
        objects = passengers + depots + walls + taxis
        fg = np.any(objects > 0, axis=-1)

        # compute background
        bg = np.ones_like(walls) * get_rgb('white')
        bg[fg, :] = 0

        # construct border
        border_color = get_rgb('white' if (not self._rgb) or not in_taxi else goal_color)
        image = generate_border(in_taxi,
                                img_shape=img_shape,
                                dash_widths=dash_widths,
                                color=border_color)

        # insert content on top of border
        content = bg + objects
        pad_top_left, pad_bot_right = banner_widths
        image[pad_top_left:-pad_bot_right, pad_top_left:-pad_bot_right, :] = content

        if (not self._rgb):
            image = np.mean(image, axis=-1, keepdims=True)

        return image.astype(np.float32)

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

def get_good_color(color):
    colorname = color
    colorname = 'gold' if colorname == 'yellow' else colorname
    colorname = 'c' if colorname == 'cyan' else colorname
    colorname = 'm' if colorname == 'magenta' else colorname
    colorname = 'silver' if colorname in ['gray', 'grey'] else colorname
    return colorname

def get_rgb(colorname):
    good_color = get_good_color(colorname)
    color_tuple = colors.hex2color(colors.get_named_colors_mapping()[good_color])
    return np.asarray(color_tuple)

def to_rgb(array):
    """Add a channel dimension with 3 entries"""
    return np.tile(array[:, :, np.newaxis], (1, 1, 3))

def cell_start(position, cell_width, wall_width):
    """Compute <row, col> indices of top-left pixel of cell at given position"""
    row, col = position
    row_start = wall_width + row * (cell_width + wall_width)
    col_start = wall_width + col * (cell_width + wall_width)
    return (row_start, col_start)

def expand_grid(grid, cell_width, wall_width):
    """Expand the built-in maze grid using the provided width information"""
    for row_or_col_axis in [0, 1]:
        slices = np.split(grid, grid.shape[row_or_col_axis], axis=row_or_col_axis)
        walls = slices[0::2]
        cells = slices[1::2]
        walls = [np.repeat(wall, wall_width, axis=row_or_col_axis) for wall in walls]
        cells = [np.repeat(cell, cell_width, axis=row_or_col_axis) for cell in cells]
        slices = [item for pair in zip(walls, cells) for item in pair] + [walls[-1]]
        grid = np.concatenate(slices, axis=row_or_col_axis).astype(float)
    return grid

def passenger_patch(cell_width, passenger_width, in_taxi):
    """Generate a patch representing a passenger, along with any associated marks"""
    assert passenger_width <= cell_width
    sw_bg = np.tri(cell_width // 2, k=(cell_width // 2 - passenger_width // 2 - 2), dtype=int)
    nw_bg = np.flipud(sw_bg)
    ne_bg = np.fliplr(nw_bg)
    se_bg = np.fliplr(sw_bg)

    bg = np.block([[nw_bg, ne_bg], [sw_bg, se_bg]])

    # add center row / column for odd widths
    if cell_width % 2 == 1:
        bg = np.insert(bg, cell_width // 2, 0, axis=0)
        bg = np.insert(bg, cell_width // 2, 0, axis=1)

    # crop edges to a circle and invert
    excess = (cell_width - passenger_width) // 2
    bg[:excess, :] = 1
    bg[:, :excess] = 1
    bg[-excess:, :] = 1
    bg[:, -excess:] = 1
    patch = (1 - bg)

    # add marks relating to 'in_taxi'
    center = cell_width // 2
    if in_taxi:
        marks = np.zeros_like(patch)
        marks[center, :] = 1
        marks[:, center] = 1
    else:
        marks = np.eye(cell_width, dtype=int) | np.fliplr(np.eye(cell_width, dtype=int))
    marks[patch == 0] = 0

    return patch, marks

def depot_patch(cell_width, depot_width):
    """Generate a patch representing a depot"""
    assert depot_width <= cell_width // 2
    sw_patch = np.tri(cell_width // 2, k=(depot_width - cell_width // 2))
    nw_patch = np.flipud(sw_patch)
    ne_patch = np.fliplr(nw_patch)
    se_patch = np.fliplr(sw_patch)

    patch = np.block([[nw_patch, ne_patch], [sw_patch, se_patch]])

    # add center row / column for odd widths
    if cell_width % 2 == 1:
        patch = np.insert(patch, cell_width // 2, 0, axis=0)
        patch = np.insert(patch, cell_width // 2, 0, axis=1)

    patch[0, :] = 1
    patch[:, 0] = 1
    patch[-1, :] = 1
    patch[:, -1] = 1

    return patch

def taxi_patch(cell_width, depot_width, passenger_width):
    """Generate a patch representing a taxi"""
    depot = depot_patch(cell_width, depot_width)
    passenger, _ = passenger_patch(cell_width, passenger_width, False)
    taxi_patch = 1 - (depot + passenger)

    # crop edges
    taxi_patch[0, :] = 0
    taxi_patch[:, 0] = 0
    taxi_patch[-1, :] = 0
    taxi_patch[:, -1] = 0

    return taxi_patch

def generate_border(in_taxi, img_shape=(84, 84), dash_widths=(6, 6), color=None):
    """Generate a border to reflect the current in_taxi status"""
    if in_taxi:
        # pad with dashes to 84x84
        dw_r, dw_c = dash_widths
        n_repeats = (img_shape[0] // (2 * dw_r), img_shape[1] // (2 * dw_c))
        image = np.tile(
            np.block([
                [np.ones((dw_r, dw_c)), np.zeros((dw_r, dw_c))],
                [np.zeros((dw_r, dw_c)), np.ones((dw_r, dw_c))],
            ]), n_repeats)

        # convert to color 84x84x3
        image = np.tile(np.expand_dims(image, -1), (1, 1, 3))
        image = image * color
    else:
        # pad with white to 84x84x3
        image = np.ones(img_shape + (3, ))

    return image
