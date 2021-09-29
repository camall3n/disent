#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import gzip
import pickle
import numpy as np
import logging


log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def get_closest_mask(usage_ratio: float, pickle_file: str) -> np.ndarray:
    """
    This function is intended to be used with the data
    generated by `run_04_gen_adversarial_ruck.py`

    The function finds the closest member in the population with
    the matching statistic. The reason this function works is that
    the population should consist only of near-pareto-optimal solutions.
    - These solutions are found using NSGA2
    """
    # load pickled data
    with gzip.open(pickle_file, mode='rb') as fp:
        data = pickle.load(fp)
        values = np.array(data['values'], dtype='bool')
        scores = np.array(data['scores'], dtype='float64')
        del data
    # check shapes
    assert values.ndim == 2
    assert scores.ndim == 2
    assert scores.shape == (len(values), 2)
    # get closest
    best_indices = np.argsort(np.abs(scores[:, 1] - usage_ratio))[:5]
    # print stats
    log.info(f'The {len(best_indices)} closest members to target usage={usage_ratio:7f}')
    for i, idx in enumerate(best_indices):
        assert np.isclose(np.mean(values[idx]), scores[idx, 1]), 'member fitness_usage is not close to the actual mask usage. The data is invalid.'
        log.info(f' [{i+1}] idx={idx:04d} overlap={scores[idx, 0]:7f} usage={scores[idx, 1]:7f}')
    # return the best!
    return values[best_indices[0]]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
