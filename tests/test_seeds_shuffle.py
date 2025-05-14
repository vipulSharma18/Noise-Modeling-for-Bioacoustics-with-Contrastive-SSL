"""
All tests to check if seed is correctly used in datasets.
Check that the dataset has different data for the same index and different seeds.
This test makes sure that cross validation results are valid.
"""
import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from stable_ssl.utils import seed_everything
from ssl_bioacoustics.custom_datasets import UrbanSoundDataset, AudioDataset


@pytest.mark.parametrize(("seeds", "dataset"), [((0, 99), UrbanSoundDataset), ((0, 99), AudioDataset)])
def test_diff_seed_same_index_diff_samples(seeds, dataset):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    seed_everything(seeds[0])
    if dataset == UrbanSoundDataset:
        dataset1 = dataset(
            root_dir='/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/',
            fold=1,
            csv_file="/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/UrbanSound8K.csv",
            transform=transform
            )
    elif dataset == AudioDataset:
        # Instantiate the dataset
        dataset1 = dataset(
            root_dir='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/',
            meta_file='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/train-clean.csv',
            split='train',
            transform=transform,
            )
    dataloader1 = iter(DataLoader(dataset1, batch_size=1, shuffle=True))
    sample1 = next(dataloader1)
    sample1 = sample1[0] if len(sample1) == 2 else sample1
    assert sample1 is not None

    seed_everything(seeds[1])
    if dataset == UrbanSoundDataset:
        dataset2 = dataset(
            root_dir='/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/',
            fold=1,
            csv_file="/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/UrbanSound8K.csv",
            transform=transform
            )
    elif dataset == AudioDataset:
        # Instantiate the dataset
        dataset2 = dataset(
            root_dir='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/',
            meta_file='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/train-clean.csv',
            split='train',
            transform=transform,
            )
    dataloader2 = iter(DataLoader(dataset2, batch_size=1, shuffle=True))
    sample2 = next(dataloader2)
    sample3 = next(dataloader2)
    sample2 = sample2[0] if len(sample2) == 2 else sample2
    sample3 = sample3[0] if len(sample3) == 2 else sample3
    assert sample2 is not None
    assert sample3 is not None

    # dataloader uses random seed for shuffling
    assert torch.allclose(sample1, sample3) is False
    assert torch.allclose(sample1, sample2) is False

    # datasets are still returning the same sample at idx 0, i.e.,
    # the seed is irrelevant for the datasets but only used in shuffling in the dataloaders
    sample_idx0_d1 = dataset1[0][0] if len(dataset1[0]) == 2 else dataset1[0]
    sample_idx0_d2 = dataset2[0][0] if len(dataset2[0]) == 2 else dataset2[0]
    assert torch.allclose(sample_idx0_d1, sample_idx0_d2) is True
