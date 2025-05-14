import torch
from torchvision.transforms import v2
from ssl_bioacoustics.custom_samplers import ConcatViewsSampler


def test_concat_view_sampler():
    transforms = [
        v2.Compose([v2.RandomHorizontalFlip()]),
        v2.Compose([v2.RandomHorizontalFlip()]),
    ]
    num_transforms = len(transforms)
    concat_sampler = ConcatViewsSampler(transforms)
    channels, height, width = 3, 32, 32
    input = torch.ones(channels, height, width)
    output = concat_sampler(input)
    assert output.shape == (num_transforms, channels, height, width)
