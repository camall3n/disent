import copy

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from disent.dataset.data import TaxiData64x64
from disent.dataset import DisentDataset
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32

data = TaxiData64x64()
dataset = DisentDataset(dataset=data, sampler=SingleSampler(), transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=2,
                        persistent_workers=True)
obs, states = dataset.dataset_sample_batch_with_factors(16, mode='input')

for ob in obs:
    plt.imshow(ob.moveaxis(0, -1))
    plt.show()
