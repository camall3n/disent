from argparse import Namespace # Needed for eval to compute markov_args
import glob
import os
import platform
import torch

from markov_abstr.gridworld.models.nnutils import Network
from markov_abstr.gridworld.models.featurenet import FeatureNet

class MarkovAbstraction(Network):
    def __init__(self, x_shape):
        super().__init__()
        assert x_shape == (3, 84, 84)

        markov_abstraction_tag = 'exp78-blast-markov_122__learningrate_0.001__latentdims_20'
        prefix = '~/data-gdk/csal/factored/' if platform.system() == 'Linux' else '~/dev/factored-reps/'
        expanded_prefix = os.path.expanduser(prefix)
        results_dir = expanded_prefix + 'results/'

        args_filename = glob.glob(results_dir +
                                'taxi/logs/{}/args-*.txt'.format(markov_abstraction_tag))[0]
        with open(args_filename, 'r') as args_file:
            markov_args = eval(args_file.read())

        self.latent_dims = markov_args.latent_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(self.device))

        self.featurenet = FeatureNet(markov_args,
                                n_actions=5,
                                input_shape=x_shape,
                                latent_dims=markov_args.latent_dims,
                                device=self.device).to(self.device)
        model_file = results_dir + 'taxi/models/{}/fnet-{}_best.pytorch'.format(
            markov_abstraction_tag, markov_args.seed)
        self.featurenet.load(model_file, to=self.device)
        self.featurenet.freeze()
        self.phi = self.featurenet.phi
        self.phi.freeze()

    def encode(self, x):
        return self.phi(x)
