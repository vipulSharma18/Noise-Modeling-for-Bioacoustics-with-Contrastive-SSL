"""
Module to check the dotpath lookup of ssl_bioacoustics class via hydra.

Relies on hydra.utils.get_class, hydra.utils.get_method, hydra.utils.get_object for dotpath lookup.
https://hydra.cc/docs/advanced/instantiate_objects/overview/#dotpath-lookup-machinery

"""


def test_utils_scheduler():
    """ssl_bioacoustics.utils.dummy_scheduler_lambda"""
    from hydra.utils import get_method
    from ssl_bioacoustics.utils import dummy_scheduler_lambda
    assert get_method("ssl_bioacoustics.utils.dummy_scheduler_lambda") == dummy_scheduler_lambda


def test_urban_sound_dataset():
    """ssl_bioacoustics.custom_datasets.urban_sound_8k.UrbanSoundDataset"""
    from hydra.utils import get_class
    from ssl_bioacoustics.custom_datasets import UrbanSoundDataset
    assert get_class("ssl_bioacoustics.custom_datasets.UrbanSoundDataset") == UrbanSoundDataset


def test_audio_dataset():
    """ssl_bioacoustics.custom_datasets.audio_dataset.AudioDataset"""
    from hydra.utils import get_class
    from ssl_bioacoustics.custom_datasets import AudioDataset
    assert get_class("ssl_bioacoustics.custom_datasets.AudioDataset") == AudioDataset


def test_frozen_noise_dataset():
    """ssl_bioacoustics.custom_datasets.frozen_noise.FrozenNoiseDataset"""
    from hydra.utils import get_class
    from ssl_bioacoustics.custom_datasets import FrozenNoiseDataset
    assert get_class("ssl_bioacoustics.custom_datasets.FrozenNoiseDataset") == FrozenNoiseDataset


def test_transform_spectrogram():
    """ssl_bioacoustics.custom_transforms.Spectrogram"""
    from hydra.utils import get_method
    from ssl_bioacoustics.custom_transforms import Spectrogram
    assert get_method("ssl_bioacoustics.custom_transforms.Spectrogram") == Spectrogram


def test_transform_chunk_and_shuffle():
    """ssl_bioacoustics.custom_transforms.ChunkShuffleLabel"""
    from hydra.utils import get_method
    from ssl_bioacoustics.custom_transforms import ChunkShuffleLabel
    assert get_method("ssl_bioacoustics.custom_transforms.ChunkShuffleLabel") == ChunkShuffleLabel
