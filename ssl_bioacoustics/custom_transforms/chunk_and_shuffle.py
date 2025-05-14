"""
Implement a custom chunk, shuffle and label transform for use
in shuffle and learn type temporal order pretraining tasks.
"""

import math
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import ToImage


class ChunkShuffleLabel(torch.nn.Module):
    """
    Note:
    -----
    Use this at the end of the transforms pipeline because it
    will add the temporal order verification label to the data and create a
    tuple breaking later transforms.

    Parameters:
    -----------
    sampling_rate: int
        The sampling rate of the audio input.
        This will be used to convert the chunk size to samples.
    chunk_size: float
        The size of each chunk in seconds.
    number_of_chunks: int
        The number of chunks to split the data into.
    chunk_order: list[int]
        The order in which to shuffle the chunks.
        Should be 0-indexed and in the range [0, number_of_chunks-1].
    label: int
        The label for this temporal order of chunks.
        It's appended to the chunks list and returned.
    """
    def __init__(
        self,
        chunk_order,
        label,
        sampling_rate=None,
        chunk_size=None,
        number_of_chunks=None,
    ):
        super(ChunkShuffleLabel, self).__init__()
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.number_of_chunks = number_of_chunks
        if self.sampling_rate is None and self.chunk_size is None and self.number_of_chunks is None:
            raise(ValueError("sampling_rate, chunk_size and number_of_chunks cannot be None together."))
        invalid = [
            i for i in chunk_order
            if (i < 0) or (i >= number_of_chunks)
            ]
        if len(invalid) > 0:
            raise ValueError(
                f"Invalid chunk order {invalid}. \
                    It should be >= 0 and < {self.number_of_chunks}."
                )
        self.chunk_order = chunk_order
        self.label = label

    def forward(self, x):
        """
        Shuffle the chunks of the input data and the labels.

        Parameters:
        -----------
        x: np.ndarray or torch.Tensor or PIL Image
            The inputs cans be (batch_sz, len_waveform),
            (batch_sz, channels, n_freq, n_time).
        """
        if isinstance(x, Image.Image):
            x = ToImage()(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            raise ValueError(
                f"Unsupported input type {type(x)}."
                )
        total_length = x.shape[-1]
        # check if sampling_rate is None and/or chunk_size is None and populate appropriately.
        # once populated, make sure that the length doesn't exceed the total length.
        if self.sampling_rate is None and self.chunk_size is None:
            self.sampling_rate = 1
            self.chunk_size = math.ceil(total_length / self.number_of_chunks)

        if self.chunk_size is None and self.sampling_rate is not None:
            self.chunk_size = math.ceil(total_length / (
                self.number_of_chunks * self.sampling_rate
                ))
        elif self.sampling_rate is None and self.chunk_size is not None:
            self.sampling_rate = math.ceil(total_length / (
                self.number_of_chunks * self.chunk_size
                ))

        # there can be either 2 or 3 dimensions in x but we only care about the last one.
        chunks = []
        for i in self.chunk_order:
            end = min(total_length, int((i + 1) * self.chunk_size * self.sampling_rate))
            start = max(0, end - int(self.chunk_size * self.sampling_rate))
            chunks.append(x[..., start:end])
        label_tensor = torch.full(chunks[0].shape, self.label)
        chunks.append(label_tensor)
        chunks = torch.stack(chunks)
        return chunks


class ShuffleLabel(torch.nn.Module):
    """
    Same as ChunkShuffleLabel but the input is already be chunked, say by a sliding window transform.
    Shuffle the chunks of the input data and the labels.
    Note:
    -----
    Use this at the end of the transforms pipeline because it
    will add the temporal order verification label to the data and create a
    tuple breaking later transforms.

    Parameters:
    -----------
    sampling_rate: int
        The sampling rate of the audio input.
        This will be used to convert the chunk size to samples.
    chunk_size: float
        The size of each chunk in seconds.
    number_of_chunks: int
        The number of chunks to split the data into.
    chunk_order: list[int]
        The order in which to shuffle the chunks.
        Should be 0-indexed and in the range [0, number_of_chunks-1].
    label: int
        The label for this temporal order of chunks.
        It's appended to the chunks list and returned.
    """
    def __init__(
        self,
        chunk_order,
        label,
        sampling_rate=None,
        chunk_size=None,
        number_of_chunks=None,
    ):
        super(ChunkShuffleLabel, self).__init__()
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.number_of_chunks = number_of_chunks
        if self.sampling_rate is None and self.chunk_size is None and self.number_of_chunks is None:
            raise(ValueError("sampling_rate, chunk_size and number_of_chunks cannot be None together."))
        invalid = [
            i for i in chunk_order
            if (i < 0) or (i >= number_of_chunks)
            ]
        if len(invalid) > 0:
            raise ValueError(
                f"Invalid chunk order {invalid}. \
                    It should be >= 0 and < {self.number_of_chunks}."
                )
        self.chunk_order = chunk_order
        self.label = label

    def forward(self, x):
        """
        Shuffle the chunks of the input data and the labels.

        Parameters:
        -----------
        x: np.ndarray or torch.Tensor or PIL Image
            The inputs cans be (batch_sz, len_waveform),
            (batch_sz, channels, n_freq, n_time).
        """
        if isinstance(x, Image.Image):
            x = ToImage()(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            raise ValueError(
                f"Unsupported input type {type(x)}."
                )
        total_length = x.shape[-1]
        # check if sampling_rate is None and/or chunk_size is None and populate appropriately.
        # once populated, make sure that the length doesn't exceed the total length.
        if self.sampling_rate is None and self.chunk_size is None:
            self.sampling_rate = 1
            self.chunk_size = math.ceil(total_length / self.number_of_chunks)

        if self.chunk_size is None and self.sampling_rate is not None:
            self.chunk_size = math.ceil(total_length / (
                self.number_of_chunks * self.sampling_rate
                ))
        elif self.sampling_rate is None and self.chunk_size is not None:
            self.sampling_rate = math.ceil(total_length / (
                self.number_of_chunks * self.chunk_size
                ))

        # there can be either 2 or 3 dimensions in x but we only care about the last one.
        chunks = []
        for i in self.chunk_order:
            end = min(total_length, int((i + 1) * self.chunk_size * self.sampling_rate))
            start = max(0, end - int(self.chunk_size * self.sampling_rate))
            chunks.append(x[..., start:end])
        label_tensor = torch.full(chunks[0].shape, self.label)
        chunks.append(label_tensor)
        chunks = torch.stack(chunks)
        return chunks
