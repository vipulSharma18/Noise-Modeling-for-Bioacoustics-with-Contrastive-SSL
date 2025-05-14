import numpy as np
import scipy
import torch
from ssl_bioacoustics.custom_transforms import Spectrogram
from ssl_bioacoustics.custom_transforms import ChunkShuffleLabel


def test_chunk_shuffle_label_raw_audio():
    """
    Test the ChunkShuffleLabel transform.
    """
    # Create a random tensor representing 1 sample
    channels, n_freq, n_time = 3, 70, 228800
    sampling_rate = 22000
    x = torch.randn(channels, n_freq, n_time)
    chunk_size = 2.5
    number_of_chunks = 4
    chunk_order = [0, 1, 2]
    label = 1
    chunk_time = int(chunk_size * sampling_rate)

    # Instantiate the transform
    transform = ChunkShuffleLabel(
        chunk_order=chunk_order,
        label=label,
        sampling_rate=sampling_rate,
        chunk_size=chunk_size,
        number_of_chunks=number_of_chunks,
        )

    # Apply the transform
    x = transform(x)

    assert torch.is_tensor(x)
    assert (len(torch.unique(x[-1])) == 1)
    assert torch.unique(x[-1]).item() == label

    # chunks are of the right length
    assert x.shape[-1] == chunk_size * sampling_rate

    # Check the shape
    assert x.shape == (
        len(chunk_order)+1,
        channels,
        n_freq,
        chunk_time
        )

    # Check the processing back to the (x,y) logic
    segments = x[:-1]
    extracted_labels = x[-1].squeeze().long()
    extracted_labels = extracted_labels.view(-1)
    extracted_labels_2 = extracted_labels[..., 0].squeeze()
    extracted_labels = torch.unique(extracted_labels, dim=-1).squeeze()
    assert torch.is_tensor(extracted_labels)
    assert torch.is_tensor(extracted_labels_2)
    assert torch.is_tensor(segments)
    assert extracted_labels == label  # it will be a scalar tensor.
    assert extracted_labels_2 == label  # test the method used in shuffle and learn which is optimal
    assert segments.shape == (
        len(chunk_order),
        channels,
        n_freq,
        chunk_time
        )


def test_chunk_shuffle_label_spectrogram():
    """
    Test the ChunkShuffleLabel transform.
    """

    sampling_rate = 22000
    n_time = 228800
    x = np.arange(n_time)
    x = scipy.signal.chirp(
        x, f0=100, f1=1000, t1=n_time//2, method='logarithmic'
        )

    spectrogram_transform = Spectrogram(
        sampling_rate=sampling_rate,
        representation='power_cqt',
        convert_to_db=False,
        representation_mode='raw'
        )

    # Apply the transform
    x = spectrogram_transform(x)

    number_of_chunks = 4
    chunk_order = [0, 1, 2]
    label = 1

    # Instantiate the transform
    transform = ChunkShuffleLabel(
        chunk_order=chunk_order,
        label=label,
        number_of_chunks=number_of_chunks,
        )

    # Apply the transform
    x = transform(x)

    assert torch.is_tensor(x)
    assert (len(torch.unique(x[-1])) == 1)
    assert torch.unique(x[-1]).item() == label

    # chunks are of the right length
    assert x.shape[0] == len(chunk_order) + 1

    # # Check the processing back to the (x,y) logic
    segments = x[:-1]
    extracted_labels = x[-1].squeeze().long()
    extracted_labels = extracted_labels.view(-1)
    extracted_labels_2 = extracted_labels[..., 0].squeeze()
    extracted_labels = torch.unique(extracted_labels, dim=-1).squeeze()
    assert torch.is_tensor(extracted_labels)
    assert torch.is_tensor(extracted_labels_2)
    assert torch.is_tensor(segments)
    assert extracted_labels == label  # it will be a scalar tensor.
    assert extracted_labels_2 == label  # test the method used in shuffle and learn which is optimal
    assert segments.shape[0] == len(chunk_order)
