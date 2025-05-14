"""
Creates overlapping chunks of the input data along the time axis.
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.v2 import ToImage


class SlidingWindowTransform(torch.nn.Module):
    """
    Note:
    -----
    This transform operates per sample, not per batch.
    Potential upgrade: add batch_first (torch.nn.utils.rnn.pad_sequence) type parameter to handle batching.

    Description:
    ------------
    This transform is used to create overlapping or non-overlapping windows of the input data along the time axis.
    This will add a new leading dimension, i.e., number of frames/slides, to the input data.
    This converts from 2D (n_freq, n_time) to 3D (n_frames, n_freq, n_time).

    Parameters:
    -----------
    sampling_rate: int
        The sampling rate of the audio input.
        This will be used to convert the window size to samples.
    window_size: float
        The size of each window in seconds.
    hop_length: float
        The size of the hop/stride between windows in seconds.
    """
    def __init__(
        self,
        sampling_rate=None,
        window_size=None,
        hop_length=0,
    ):
        super(SlidingWindowTransform, self).__init__()
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.hop_length = hop_length
        if self.sampling_rate is None or self.window_size is None:
            raise(ValueError("sampling_rate, window_size cannot be None."))

    def forward(self, x):
        """
        Slice the input data along the time axis with overlapping windows.

        Parameters:
        -----------
        x: np.ndarray or torch.Tensor
            The inputs can be (len_waveform) or (channels, n_time),
            the windowing will be applied along the last dimension.

        Returns:
        --------
        torch.Tensor
            A 3D tensor with the shape (n_frames, self.window_size*self.sampling_rate), or, in general (n_frames, ..., self.window_size*self.sampling_rate).
        """

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            raise ValueError(
                f"Unsupported input type {type(x)}."
                )

        total_length = x.shape[-1]//self.sampling_rate  # in seconds
        leading_dims_x = x.shape[:-1]
        # reference: https://github.com/microsoft/bird-acoustics-rcnn/blob/main/process_data.ipynb
        # Kahl, S. et al. (2021) ‘BirdNET: A deep learning solution for avian diversity monitoring’, Ecological Informatics, 61, p. 101236. Available at: https://doi.org/10.1016/j.ecoinf.2021.101236.
        num_windows_max_possible = max(1, 1 + int((total_length - self.window_size)/self.hop_length))  # this avoids padded entries for the last window, unless there's only 1 incomplete window.

        # there can be either 2 or 3 dimensions in x but we only care about the last one.
        slides = torch.empty(num_windows_max_possible, *leading_dims_x, int(self.window_size*self.sampling_rate))
        for wind_ind in range(num_windows_max_possible):
            start = wind_ind*int(self.hop_length*self.sampling_rate)
            end = start + int(self.window_size*self.sampling_rate)
            pad_length = max(0, (end - start) - max(total_length*self.sampling_rate - start, 0))
            slides[wind_ind, ..., :] = torch.nn.functional.pad(x[..., start:end], value=0, pad=(0, pad_length))
        return slides


class PreprocessSlidingWindowWithLabels(torch.nn.Module):
    """
    Note:
    -----
    This transform is used to create overlapping chunks of the input data along the time axis.
    The output of this transform is multiple samples, each with its own label,
    as opposed to SlidingWindowTransform which outputs a single sample with multiple windows.
    Preferably use this at the beginning of the transforms pipeline.

    Parameters:
    -----------
    sampling_rate: int
        The sampling rate of the audio input.
        This will be used to convert the chunk size to samples.
    window_size: float
        The size of each window in seconds.
    hop_length: float
        The size of the hop/stride between windows in seconds.
    global_total_length: float, optional
        If you want the dataset to be considered as having a same total length,
        instead of individual length of each sample.
    """
    def __init__(
        self,
        sampling_rate=None,
        window_size=None,
        hop_length=0,
        global_total_length=None,
    ):
        super(PreprocessSlidingWindowWithLabels, self).__init__()
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.global_total_length = global_total_length
        if self.sampling_rate is None or self.window_size is None:
            raise(ValueError("sampling_rate, window_size cannot be None."))
        if self.global_total_length is not None:
            raise(Warning(f"Using global_total_length of {self.global_total_length} for all samples."))
            if self.global_total_length < self.window_size:
                raise(
                    ValueError(
                        f"global_total_length ({self.global_total_length}) must be greater than or equal to window_size ({self.window_size})."
                        )
                    )

    def forward(self, x, y=None):
        """
        Slice the input data along the time axis with overlapping windows.
        Repeat the label for each window.

        Parameters:
        -----------
        x: np.ndarray or torch.Tensor or PIL Image
            The inputs cans be (len_waveform),
            (channels, n_freq, n_time), or any other shape,
            the windowing will be applied along the last dimension.
        y: np.ndarray or torch.Tensor, optional
            The labels for the input data.
        """
        if isinstance(x, Image.Image):
            x = ToImage()(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            raise ValueError(
                f"Unsupported input type {type(x)}."
                )

        if y is None:
            y = torch.empty(0)

        sample_length = x.shape[-1]//self.sampling_rate  # in seconds
        if self.global_total_length is not None:
            total_length = self.global_total_length  # indexing works fine with torch tensors, even if we overflow.
        else:
            total_length = sample_length

        leading_dims_x = x.shape[:-1]
        # reference: https://github.com/microsoft/bird-acoustics-rcnn/blob/main/process_data.ipynb
        # Kahl, S. et al. (2021) ‘BirdNET: A deep learning solution for avian diversity monitoring’,
        # Ecological Informatics, 61, p. 101236. Available at: https://doi.org/10.1016/j.ecoinf.2021.101236.
        num_windows_max_possible = max(1, int((total_length - self.window_size)/self.hop_length) + 1)

        # there can be either 2 or 3 dimensions in x but we only care about the last one.
        slides = torch.empty(num_windows_max_possible, *leading_dims_x, self.window_size*self.sampling_rate)
        labels = torch.empty(num_windows_max_possible)
        for wind_ind in range(num_windows_max_possible):
            start = wind_ind*int(self.hop_length*self.sampling_rate)
            end = start + int(self.window_size*self.sampling_rate)
            # when inside the sample, pad 0. when outside, pad the difference.
            # when partially outside, pad the difference - the sample part within the window.
            pad_length = max(0, (end - start) - max(sample_length*self.sampling_rate - start, 0))
            slides[wind_ind, ..., :] = torch.nn.functional.pad(x[..., start:end], value=0, pad=(0, pad_length))
            labels[wind_ind] = y
        return slides, labels


class PreprocessSlidingWindowMetadata(torch.nn.Module):
    """
    Description:
    ------------
    This transform is used to support dynamic preprocessing using sliding windows
    by preparing the metadata for each window before the actual sliding window transform.

    In the scenario where the input data is to be split into multiple windows, each with its own label,
    this transform can be used to expand the metadata file for each sample into multiple rows,
    which would allow the get_item function and other functions based on length of dataset to work correctly.

    Parameters:
    -----------
    window_size: float
        The size of each window in seconds.
    hop_length: float
        The size of the hop/stride between windows in seconds.
    individual_length_column_name: str, optional
        The name of the column in the metadata that contains the length of each sample.
        If provided, the dataset will have individual lengths for each sample.
        Only one of individual_length_column_name or global_total_length can be provided, and both cannot be None.
    global_total_length: float, optional
        The total length of the audio in seconds, across all the samples.
        If provided, the dataset will be considered to have a single total length,
        instead of individual length of each sample.
        Only one of individual_length_column_name or global_total_length can be provided, and both cannot be None.
    """
    def __init__(
        self,
        window_size,
        hop_length=0,
        global_total_length=None,
        individual_length_column_name="duration",
    ):
        super(PreprocessSlidingWindowMetadata , self).__init__()
        self.window_size = window_size
        self.hop_length = hop_length
        self.global_total_length = global_total_length
        if self.global_total_length is not None:
            if self.global_total_length < self.window_size:
                raise(
                    ValueError(
                        f"global_total_length ({self.global_total_length}) must be greater than or equal to window_size ({self.window_size})."
                        )
                    )
            self.num_windows_max_possible = int((self.global_total_length - self.window_size)/self.hop_length) + 1
        else:
            self.num_windows_max_possible = None
        self.individual_length_column_name = individual_length_column_name
        if not (self.individual_length_column_name is None) ^ (self.global_total_length is None):
            raise(ValueError("individual_length_column_name and global_total_length cannot both be provided, and cannot be both None."))

    def forward(self, metadata, preprocess_idx_column="preprocess_index"):
        """
        Takes the metadata dataframe, and transforms it to
        replace each sample by the number of windows it is split into.
        A column named "preprocess_index" is added to the metadata,
        which contains the index of the window in the original audio.

        Parameters:
        -----------
        metadata: pd.DataFrame
            The metadata dataframe.
        preprocess_idx_column: str, optional
            The column name of the preprocess index in the meta file. Default is "preprocess_index".
        """

        # Create a list to store the repeated rows
        repeated_rows = []

        # For each row in the metadata
        for _, row in metadata.iterrows():
            if self.individual_length_column_name is not None:
                num_windows_max_possible = int((float(row[self.individual_length_column_name]) - self.window_size)/self.hop_length) + 1
            else:
                num_windows_max_possible = self.num_windows_max_possible
            num_windows_max_possible = max(num_windows_max_possible, 1)  # don't skip the sample if it's too short to be split.
            # Repeat the row num_windows times and add window index
            for window_idx in range(num_windows_max_possible):
                new_row = row.copy()
                new_row[preprocess_idx_column] = window_idx
                repeated_rows.append(new_row)

        # Create new dataframe from repeated rows
        new_metadata = pd.DataFrame(repeated_rows)
        # Reset index to have continuous integer indices
        new_metadata = new_metadata.reset_index(drop=True)

        return new_metadata
