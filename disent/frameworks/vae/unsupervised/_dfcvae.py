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

from dataclasses import dataclass
from numbers import Number
from typing import Any
from typing import Dict
from typing import List, Optional, Union
from typing import Sequence
from typing import Tuple

import kornia
import torch
from torch import Tensor
from torchvision.models import vgg19_bn
from torch.nn import functional as F

from disent.frameworks.helper.reductions import get_mean_loss_scale
from disent.frameworks.vae.unsupervised import BetaVae
from disent.transform.functional import check_tensor


# ========================================================================= #
# Dfc Vae                                                                   #
# ========================================================================= #


class DfcVae(BetaVae):
    """
    Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291
    - Uses features generated from a pretrained model as the loss.

    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE


    Difference:
        1. MSE loss changed to BCE or MSE loss
        2. Mean taken over (batch for sum of pixels) not mean over (batch & pixels)
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(BetaVae.cfg):
        feature_layers: Optional[List[Union[str, int]]] = None

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # make dfc loss
        # TODO: this should be converted to a reconstruction loss handler that wraps another handler
        self._dfc_loss = DfcLossModule(feature_layers=self.cfg.feature_layers)

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_recon_loss(self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute ave reconstruction loss
        pixel_loss = self.recon_handler.compute_ave_loss(xs_partial_recon, xs_targ)  # (DIFFERENCE: 1)
        # compute ave deep features loss
        feature_loss = torch.stack([
            self._dfc_loss.compute_loss(self.recon_handler.activate(x_partial_recon), x_targ, reduction=self.cfg.loss_reduction)
            for x_partial_recon, x_targ in zip(xs_partial_recon, xs_targ)
        ]).mean(dim=-1)
        # reconstruction error
        # TODO: not in reference implementation, but terms should be weighted
        # TODO: not in reference but feature loss is not scaled properly
        recon_loss = (pixel_loss + feature_loss) * 0.5
        # return logs
        return recon_loss, {
            'pixel_loss': pixel_loss,
            'feature_loss': feature_loss,
        }


# ========================================================================= #
# Helper Loss                                                               #
# ========================================================================= #


class DfcLossModule(torch.nn.Module):
    """
    Loss function for the Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291

    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE

    Difference:
    - normalise data as torchvision.models require.
    """

    def __init__(self, feature_layers: Optional[List[Union[str, int]]] = None):
        """
        :param feature_layers: List of string of IDs of feature layers in pretrained model
        """
        super().__init__()
        # feature layers to use
        self.feature_layers = set(['14', '24', '34', '43'] if (feature_layers is None) else [str(l) for l in feature_layers])
        # feature network
        self.feature_network = vgg19_bn(pretrained=True)
        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False
        # Evaluation Mode
        self.feature_network.eval()

    @property
    def num(self):
        return len(self.feature_layers)

    def __call__(self, x_recon, x_targ):
        return self.compute_loss(x_recon, x_targ)

    def compute_loss(self, x_recon, x_targ, reduction='mean'):
        """
        x_recon and x_targ data should be an unnormalized RGB batch of
        data [B x C x H x W] in the range [0, 1].
        """
        features_recon = self._extract_features(x_recon)
        features_targ = self._extract_features(x_targ)
        # compute losses
        # TODO: not in reference implementation, but consider calculating mean feature loss rather than sum
        feature_loss = 0.0
        for (f_recon, f_targ) in zip(features_recon, features_targ):
            feature_loss += F.mse_loss(f_recon, f_targ, reduction='mean')
        # scale the loss accordingly
        # (DIFFERENCE: 2)
        return feature_loss * get_mean_loss_scale(x_targ, reduction=reduction)

    def _extract_features(self, inputs: Tensor) -> List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param inputs: (Tensor) [B x C x H x W] unnormalised in the range [0, 1].
        :return: List of the extracted features
        """
        # This adds inefficiency but I guess is needed...
        check_tensor(inputs, low=0, high=1, dtype=None)
        # normalise: https://pytorch.org/docs/stable/torchvision/models.html
        result = kornia.normalize(inputs, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        # calculate all features
        features = []
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if key in self.feature_layers:
                features.append(result)
        return features


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
