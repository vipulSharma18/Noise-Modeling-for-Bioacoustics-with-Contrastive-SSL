"""
Test the AudioDataset class.
"""

from ssl_bioacoustics.custom_datasets import AudioDataset
from ssl_bioacoustics.custom_transforms import Spectrogram


def test_index_reproducibility():
    """
    Access the same index twice to see that the same sample is returned.
    Also test that different samples are returned for different indices.
    """

    transforms_audio = Spectrogram(
        sampling_rate=22000,
        representation='power_mel',
        convert_to_db=True,
        representation_mode="rgb"
        )

    # Instantiate the dataset
    dataset = AudioDataset(
        root_dir='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/',
        meta_file='/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/train-clean.csv',
        split='train',
        sampling_rate=22000,
        length=None,
        transform=transforms_audio,
    )
    assert len(dataset) > 0

    image1, _ = dataset[2]
    image2, _ = dataset[2]

    image1.save('/users/vsharm44/projects/ssl-bioacoustics/logs/figures/audio_mel1.png')
    image2.save('/users/vsharm44/projects/ssl-bioacoustics/logs/figures/audio_mel2.png')

    image3, _ = dataset[len(dataset)-5]

    image3.save('/users/vsharm44/projects/ssl-bioacoustics/logs/figures/audio_mel3.png')
