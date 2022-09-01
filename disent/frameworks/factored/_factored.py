from argparse import Namespace # Needed for eval to compute markov_args
import glob
import os
import platform
import torch

from factored_rl.models.nnutils import Network
from factored_rl.models.factored.cae import CAENet
from ..markov import MarkovAbstraction

class FactoredModel(Network):
    def __init__(self, x_shape, seed, tag):
        super().__init__()

        self.markov = MarkovAbstraction(x_shape)

        # tag = 'exp21-cae-best__lr_0.001__Lrecz_0.1__distmode_mse__Lfoc_0.003__latentdims_10__Lrecxaug_0.0'
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_0.1__distmode_mse__Lfoc_0.003__latentdims_10__Lrecxaug_1.0'
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_0.1__distmode_mse__Lfoc_0.0__latentdims_10__Lrecxaug_0.0' ##
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_0.1__distmode_mse__Lfoc_0.0__latentdims_10__Lrecxaug_1.0'
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_1.0__distmode_mse__Lfoc_0.003__latentdims_10__Lrecxaug_0.0'
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_1.0__distmode_mse__Lfoc_0.003__latentdims_10__Lrecxaug_1.0'
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_1.0__distmode_mse__Lfoc_0.0__latentdims_10__Lrecxaug_0.0'  ##
        # tag = 'exp21-cae-best__lr_0.001__Lrecz_1.0__distmode_mse__Lfoc_0.0__latentdims_10__Lrecxaug_1.0'

        prefix = '~/data-gdk/csal/factored/' if platform.system() == 'Linux' else '~/dev/factored-reps/' # yapf: disable
        expanded_prefix = os.path.expanduser(prefix)
        results_dir = expanded_prefix + 'results/'

        args_filename = glob.glob(results_dir + f'focused-taxi/logs/{tag}/args-{seed}.txt')[0]
        with open(args_filename, 'r') as args_file:
            args = eval(args_file.read())

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(self.device))

        self.facnet = CAENet(args,
                             n_actions=5,
                             n_input_dims=self.markov.latent_dims,
                             n_latent_dims=args.latent_dims,
                             device=self.device).to(self.device)
        model_file = results_dir + 'focused-taxi/models/{}/focused-autoenc-{}_best.pytorch'.format(
            tag, args.seed)
        self.facnet.load(model_file, to=self.device)
        self.facnet.freeze()

    def encode(self, x):
        z = self.markov.encode(x)
        z_fac = self.facnet.encode(z)
        return z_fac
