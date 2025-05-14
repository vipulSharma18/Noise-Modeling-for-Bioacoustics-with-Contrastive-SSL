import pytest
import numpy as np
import scipy
from PIL import Image
from ssl_bioacoustics.custom_transforms import Spectrogram


@pytest.mark.parametrize(
    ("representation", "representation_mode", "convert_to_db"),
    [
        # ('amplitude_mel', 'raw', False),
        # ('amplitude_mel', 'grayscale', False),
        # ('amplitude_mel', 'rgb', False),
        # ('amplitude_mel', 'raw', True),
        # ('amplitude_mel', 'grayscale', True),
        # ('amplitude_mel', 'rgb', True),
        # ('power_mel', 'raw', False),
        # ('power_mel', 'grayscale', False),
        # ('power_mel', 'rgb', False),
        # ('power_mel', 'raw', True),
        # ('power_mel', 'grayscale', True),
        ('power_mel', 'rgb', True),
        ('complex_mel', 'raw', False),
        # ('amplitude_stft', 'raw', False),
        # ('amplitude_stft', 'grayscale', False),
        # ('amplitude_stft', 'rgb', False),
        # ('amplitude_stft', 'raw', True),
        # ('amplitude_stft', 'grayscale', True),
        # ('amplitude_stft', 'rgb', True),
        # ('power_stft', 'raw', False),
        # ('power_stft', 'grayscale', False),
        # ('power_stft', 'rgb', False),
        # ('power_stft', 'raw', True),
        # ('power_stft', 'grayscale', True),
        ('power_stft', 'rgb', True),
        ('complex_stft', 'raw', False),
        # ('amplitude_cqt', 'raw', False),
        # ('amplitude_cqt', 'grayscale', False),
        # ('amplitude_cqt', 'rgb', False),
        # ('amplitude_cqt', 'raw', True),
        # ('amplitude_cqt', 'grayscale', True),
        # ('amplitude_cqt', 'rgb', True),
        # ('power_cqt', 'raw', False),
        # ('power_cqt', 'grayscale', False),
        # ('power_cqt', 'rgb', False),
        # ('power_cqt', 'raw', True),
        # ('power_cqt', 'grayscale', True),
        ('power_cqt', 'rgb', True),
        ('complex_cqt', 'raw', False),
        # ('mfcc', 'raw', True),
        # ('mfcc', 'grayscale', True),
        ('mfcc', 'rgb', True),
        # ('amplitude_cwt', 'raw', False),
        # ('amplitude_cwt', 'grayscale', False),
        # ('amplitude_cwt', 'rgb', False),
        # ('amplitude_cwt', 'raw', True),
        # ('amplitude_cwt', 'grayscale', True),
        # ('amplitude_cwt', 'rgb', True),
        # ('power_cwt', 'raw', False),
        # ('power_cwt', 'grayscale', False),
        # ('power_cwt', 'rgb', False),
        # ('power_cwt', 'raw', True),
        # ('power_cwt', 'grayscale', True),
        # ('power_cwt', 'rgb', True),
        ]
    )
def test_spectrogram(representation, representation_mode, convert_to_db):
    """
    Test the Spectrogram transform.
    """
    sampling_rate = 22000
    n_time = 228800
    x = np.arange(n_time)
    x = scipy.signal.chirp(
        x, f0=100, f1=1000, t1=n_time//2, method='logarithmic'
        )

    spectrogram_transform = Spectrogram(
        sampling_rate=sampling_rate,
        representation=representation,
        convert_to_db=convert_to_db,
        representation_mode=representation_mode,
        )
    # Apply the transform
    x = spectrogram_transform(x)

    if representation_mode == 'raw':
        assert len(x.shape) == 3
    else:
        assert isinstance(x, Image.Image)
        assert x.mode == 'L' if representation_mode == 'grayscale' else 'RGB'
        x.save(f'logs/figures/test_representation_{representation}_{representation_mode}_db{convert_to_db}.png')
