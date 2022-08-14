from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import freeze_support

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.data import TaxiData64x64, TaxiData84x84, TaxiOracleData
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.frameworks.ae import Ae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, DecoderIdentity
from disent.model.ae import EncoderConv64, EncoderIdentity
from disent.schedule import CyclicSchedule
from disent.frameworks.markov import MarkovAbstraction
from disent.frameworks.factored import FactoredModel

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--oracle', action='store_true',
        help='Disables observation function and returns ground-truth state instead')
    parser.add_argument('--model', type=str, default='betavae',
        choices=['betavae', 'identity', 'markov', 'factored'],
        help='The type of representation model to evaluate')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # create the dataset & dataloaders
    # - ToImgTensorF32 transforms images from numpy arrays to tensors and performs checks
    # - if you use `num_workers != 0` in the DataLoader, the make sure to
    #   wrap `trainer.fit` with `if __name__ == '__main__': ...`
    if args.oracle:
        data = TaxiOracleData()
    elif args.model in ['markov', 'factored']:
        data = TaxiData84x84()
    else:
        data = TaxiData64x64()
    dataset = DisentDataset(dataset=data, sampler=SingleSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=4)

    if args.model == 'betavae':
        # create the BetaVAE model
        # - adjusting the beta, learning rate, and representation size.
        module = BetaVae(
            model=AutoEncoder(
                # z_multiplier is needed to output mu & logvar when parameterising normal distribution
                encoder=EncoderConv64(x_shape=data.x_shape, z_size=10, z_multiplier=2),
                decoder=DecoderConv64(x_shape=data.x_shape, z_size=10),
            ),
            cfg=BetaVae.cfg(
                optimizer='adam',
                optimizer_kwargs=dict(lr=1e-3),
                loss_reduction='mean_sum',
                beta=4,
            )
        )
        # cyclic schedule for target 'beta' in the config/cfg. The initial value from the
        # config is saved and multiplied by the ratio from the schedule on each step.
        # - based on: https://arxiv.org/abs/1903.10145
        module.register_schedule(
            'beta', CyclicSchedule(
                period=1024,  # repeat every: trainer.global_step % period
            )
        )

        # train model
        # - for 2048 batches/steps
        trainer = pl.Trainer(
            max_steps=2048, gpus=1 if torch.cuda.is_available() else None, logger=False
        )
        trainer.fit(module, dataloader)

    elif args.model == 'identity':
        assert args.oracle
        module = Ae(
            model=AutoEncoder(
                encoder=EncoderIdentity(x_shape=data.x_shape),
                decoder=DecoderIdentity(x_shape=data.x_shape),
            )
        )
    elif args.model == 'markov':
        # Load pre-trained Markov abstraction
        module = MarkovAbstraction(x_shape=data.x_shape)
    elif args.model == 'factored':
        # Load pre-trained Markov abstraction + factored autoenc
        module = FactoredModel(x_shape=data.x_shape, seed=args.seed)
    else:
        raise NotImplementedError()

    # compute disentanglement metrics
    # - we cannot guarantee which device the representation is on
    # - this will take a while to run
    get_repr = lambda x: module.encode(x.to(module.device))

    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
        **metric_mig(dataset, get_repr, num_train=2000),
    }

    # evaluate
    print('model:', args.model)
    print('metrics:', metrics)

if __name__ == '__main__':
    freeze_support() # do this to make sure multiprocessing works correctly
    main()
