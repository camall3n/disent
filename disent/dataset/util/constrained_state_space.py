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

from functools import lru_cache
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from disent.dataset.util.state_space import StateSpace, NonNormalisedFactorIdxs, NonNormalisedFactors
from disent.util.visualize.vis_util import get_idx_traversal


# ========================================================================= #
# Constrained Space                                                         #
# ========================================================================= #


class ConstrainedStateSpace(StateSpace):
    """
    State space where an index corresponds to coordinates (factors/positions) in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.

    Additionally, various coordinates in the factor product space can be disallowed, resulting in
    a set of valid indices that might be smaller than the product of the individual factor sizes.

    :param is_valid_fn: callable for checking whether a given position in the state space
                        satisfies the desired constraints.
    """

    def __init__(self,
                 factor_sizes: Sequence[int],
                 factor_names: Optional[Sequence[str]] = None):
        super().__init__(factor_sizes, factor_names)
        self._constrain_indices() # modifies self.__size

    def _constrain_indices(self):
        def is_valid_idx(idx):
            return self._is_valid_pos(super().idx_to_pos(idx))
        self.valid_orig_indices = [idx for idx in range(len(self)) if is_valid_idx(idx)]
        self.constrained_indices = {
            orig: new for new, orig in enumerate(self.valid_orig_indices)
        }
        self._size = len(self.valid_orig_indices)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Overrides                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    def _is_valid_pos(self, pos) -> bool:
        raise NotImplementedError

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Coordinate Transform - any dim array, only last axis counts!          #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def pos_to_idx(self, positions) -> np.ndarray:
        """
        Check that the supplied position(s) satisfy the constraints, apply original factor
        conversion math to obtain indices, then convert to constrained indices
        """
        for pos in positions:
            if not self._is_valid_pos(pos):
                raise ValueError(f'{pos} is not a valid position in the constrained state space')
        orig_indices = super().pos_to_idx(positions)
        return [self.constrained_indices[idx] for idx in orig_indices]


    def idx_to_pos(self, indices) -> np.ndarray:
        """
        Convert constrained index/indices to original index/indices, then apply original
        factor conversion math to obtain position(s)
        """
        orig_indices = [self.valid_orig_indices[idx] for idx in indices]
        return super().idx_to_pos(orig_indices)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling Functions - any dim array, only last axis counts!            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def sample_factors(self, size=None, f_idxs: Optional[NonNormalisedFactorIdxs] = None) -> np.ndarray:
        raise NotImplementedError(f'sample_factors not implemented for constrained state spaces')

    def sample_missing_factors(self, known_factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs) -> np.ndarray:
        raise NotImplementedError(f'sample_missing_factors not implemented for constrained state spaces')

    def resample_other_factors(self, factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs) -> np.ndarray:
        raise NotImplementedError(f'resample_other_factors not implemented for constrained state spaces')

    def resample_given_factors(self, factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs):
        raise NotImplementedError(f'resample_given_factors not implemented for constrained state spaces')

    def _get_f_idx_and_factors_and_size(
        self,
        f_idx: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        num: Optional[int] = None,
    ):
        raise NotImplementedError(f'_get_f_idx_and_factors_and_size not implemented for constrained state spaces')

    def sample_random_factor_traversal(
        self,
        f_idx: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        num: Optional[int] = None,
        mode: str = 'interval',
        start_index: int = 0,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(f'sample_random_factor_traversal not implemented for constrained state spaces')


    def sample_random_factor_traversal_grid(
        self,
        num: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        mode: str = 'interval',
        factor_indices: Optional[NonNormalisedFactorIdxs] = None,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(f'sample_random_factor_traversal_grid not implemented for constrained state spaces')
