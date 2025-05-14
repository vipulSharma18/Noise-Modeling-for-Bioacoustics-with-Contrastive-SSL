import pytest
import numpy as np
import scipy
from PIL import Image
from ssl_bioacoustics.custom_datasets import UrbanSoundDataset
from ssl_bioacoustics.custom_transforms import Spectrogram, EnvironmentalNoise
from .common_test_utils import generate_time_varying_sawtooth_wave


@pytest.mark.parametrize(
    ("representation", "noise_source", "classes"),
    [
        ('power_mel', UrbanSoundDataset, 'engine_idling'),
        ('power_stft', UrbanSoundDataset, 'engine_idling'),
        ('power_cqt', UrbanSoundDataset, 'engine_idling'),
        ('mfcc', UrbanSoundDataset, 'engine_idling'),
        # ('power_cwt', UrbanSoundDataset, 'engine_idling'),
        ]
    )
def test_natural_corruption(representation, noise_source, classes):
    """
    Test the output of natural corruptions.
    """

    sampling_rate = 22000
    x = generate_time_varying_sawtooth_wave(
        duration=10,
        sampling_rate=sampling_rate)

    spectrogram_transform = Spectrogram(
        sampling_rate=sampling_rate,
        representation=representation,
        convert_to_db=True,
        representation_mode='rgb',
        )
    # Apply the transform
    clean_spec = spectrogram_transform(x)

    assert isinstance(clean_spec, Image.Image)
    clean_spec.save(f'logs/figures/test_corruption_clean_{representation}.png')

    natural_noise = EnvironmentalNoise(
        sampling_rate=sampling_rate,
        noise_source=noise_source,
        classes=classes,
        snr=10,
        csv_file='data/UrbanSound8k/metadata/UrbanSound8K.csv',
        root_dir='data/UrbanSound8k/audio',
        )

    corrupted_x = natural_noise(x)
    corrupted_spec = spectrogram_transform(corrupted_x)

    assert isinstance(corrupted_spec, Image.Image)
    corrupted_spec.save(f'logs/figures/test_corruption_noisy_{representation}_{classes}.png')
