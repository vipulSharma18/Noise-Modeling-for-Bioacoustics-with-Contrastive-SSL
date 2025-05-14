"""
Test that same corruption is generated with
same index irrespective of the global seed.
"""
import pytest
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from stable_ssl.utils import seed_everything
from ssl_bioacoustics.custom_transforms import EnvironmentalNoise, Spectrogram
from ssl_bioacoustics.custom_datasets import UrbanSoundDataset, AudioDataset
from ssl_bioacoustics.custom_datasets import FrozenNoiseDataset


@pytest.mark.parametrize(("seeds", "dataset"), [
    ((100, 1), UrbanSoundDataset),
    ((100, 1), AudioDataset)
    ])
def test_diff_seed_same_index_same_corruption(seeds, dataset):
    """
    Corruption depends on the index of the sample and hence should be same.
    """
    spectrogram_transform = Spectrogram(
        sampling_rate=22000,
        representation='power_mel',
        convert_to_db=True,
        representation_mode='rgb',
        )
    transform = transforms.Compose([
        spectrogram_transform,
    ])
    noise_transform = EnvironmentalNoise(
        sampling_rate=22000,
        noise_source=UrbanSoundDataset,
        classes='jackhammer',
        snr=-5,
        csv_file='data/UrbanSound8k/metadata/UrbanSound8K.csv',
        root_dir='data/UrbanSound8k/audio',
        )
    combined_transform = transforms.Compose([
        noise_transform,
        transform,
    ])
    if dataset == UrbanSoundDataset:  # we're using the audio version of urban sound, not the images
        kwargs = {
            'root_dir': 'data/UrbanSound8k/audio',
            'csv_file': 'data/UrbanSound8k/metadata/UrbanSound8K.csv',
        }
    elif dataset == AudioDataset:
        kwargs = {
            'root_dir': 'data/Birdsong/audio/raw-audio-unfiltered-22000-sample/',
            'meta_file': 'data/Birdsong/audio/raw-audio-unfiltered-22000-sample/train-clean.csv',
            'split': 'train',
        }
    seed_everything(seeds[0])
    frozen_noise_dataset = FrozenNoiseDataset(
        dataset=dataset,
        noise_transform=noise_transform,
        transform=transform,
        **kwargs)
    sample1, _ = frozen_noise_dataset[0]
    vanilla_dataset = dataset(transform=combined_transform, **kwargs)
    null_h_sample1, _ = vanilla_dataset[0]

    seed_everything(seeds[1])
    frozen_noise_dataset = FrozenNoiseDataset(
        dataset=dataset,
        noise_transform=noise_transform,
        transform=transform,
        **kwargs)
    sample2, _ = frozen_noise_dataset[0]
    vanilla_dataset = dataset(transform=combined_transform, **kwargs)
    null_h_sample2, _ = vanilla_dataset[0]

    assert sample1 is not None
    assert sample2 is not None
    assert null_h_sample1 is not None
    assert null_h_sample2 is not None

    # POSITIVE Hypothesis: Frozen noise works
    # since dataset is independent of seed (only dataloader uses it for shuffling)
    # the same index will have same data. If the corruption is same, the samples
    # should be same and our test passes.
    s1_tensor = transforms.ToImage()(sample1)
    s2_tensor = transforms.ToImage()(sample2)
    assert torch.allclose(s1_tensor, s2_tensor)
    assert isinstance(sample1, Image.Image)
    sample1.save('logs/figures/test_frozen_noise_sample1_positive.png')
    assert isinstance(sample2, Image.Image)
    sample2.save('logs/figures/test_frozen_noise_sample2_positive.png')

    # NULL Hypothesis: Even without frozen noise, we get the same samples
    s1_tensor = transforms.ToImage()(null_h_sample1)
    s2_tensor = transforms.ToImage()(null_h_sample2)
    assert not torch.allclose(s1_tensor, s2_tensor)
    assert isinstance(null_h_sample1, Image.Image)
    null_h_sample1.save('logs/figures/test_frozen_noise_sample1_null.png')
    assert isinstance(null_h_sample2, Image.Image)
    null_h_sample2.save('logs/figures/test_frozen_noise_sample2_null.png')
