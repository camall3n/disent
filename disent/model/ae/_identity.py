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

import numpy as np
import torch
from torch import nn, float32
from torch import Tensor

from disent.model import DisentDecoder
from disent.model import DisentEncoder


# ========================================================================= #
# test models                                                               #
# ========================================================================= #

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        x.requires_grad_()
        return x * 1.0


class EncoderIdentity(DisentEncoder):

    def __init__(self, x_shape=(3, 64, 64)):
        super().__init__(x_shape=x_shape, z_size=int(np.prod(x_shape)), z_multiplier=1)

        self.model = nn.Sequential(
            Identity(),
            nn.Flatten()
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderIdentity(DisentDecoder):

    def __init__(self, x_shape=(3, 64, 64)):
        super().__init__(x_shape=x_shape, z_size=int(np.prod(x_shape)), z_multiplier=1)

        self.model = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=self.x_shape),
            nn.Identity(),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
