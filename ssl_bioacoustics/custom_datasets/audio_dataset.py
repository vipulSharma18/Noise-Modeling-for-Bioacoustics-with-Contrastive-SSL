"""
Reusable audio dataset which supports:
- parameterized chunking of the audio data.
- usage of spectrograms.
- dynamic preprocessing.
- caching of the dataset after preprocessing, within the lifetime of the dataset object.
- reuse of preprocessed data if it exists, across different dataset objects.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import os
from ssl_bioacoustics.utils import isolate_rng


class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        meta_file,
        transform=None,
        split="train",
        sampling_rate=None,
        length=None,
        offset=0,
        pad_token=0,
        data_column="sample",
        label_column="label",
        preprocess_fn=None,
        preprocess_metadata_fn=None,
        preprocessed_cache_dir=None,
        preprocess_idx_column = 'preprocess_index',
        reuse_preprocessed_data=False,
        enable_inmemory_cache=False,
    ):
        """
        The meta file can be used to filter the samples in the dataset.
        Make sure to use "sample" column name.
        It may also contain a "label" column if available in the data.

        Parameters:
        -----------
        root_dir: str
            The root directory of the dataset.
        meta_file: str
            The path to the meta file containing samples to filter and optional labels.
        transform: list[torch.nn.Module], optional
            The transforms to apply to the audio data.
        split: str, optional
            The split to use. It will be appended to the root_dir to form the path to the audio files. Default is "train".
        sampling_rate: int, optional
            The sampling rate of the audio data. Default is None, which uses the file's sampling rate.
        length: float, optional
            If not None, the audio will be truncated or tiled to this length (in seconds).
        offset: float, optional
            If not None, the audio will be offset by this amount (in seconds).
        pad_token: float, optional
            The token to use for padding the audio. Default is 0.
        data_column: str, optional
            The column name of the data in the meta file. Default is "sample".
        label_column: str, optional
            The column name of the label in the meta file. Default is "label".
        preprocess_fn: callable, optional
            A function to apply to the audio data and labels before any transformations.
        preprocess_metadata_fn: callable, optional
            A function to apply to the metadata to allow for dynamic preprocessing.
        preprocessed_cache_dir: str, optional
            The directory to cache the preprocessed data.
        preprocess_idx_column: str, optional
            The column name of the preprocess index in the meta file. Default is "preprocess_index".
        reuse_preprocessed_data: bool, optional
            If True, the preprocessed data will be reused if it exists. Default is False.
        enable_inmemory_cache: bool, optional
            If True, the preprocessed data will be cached in memory. Default is False.
        """
        self.root_dir = root_dir
        self.split = split
        self.preprocessed_cache_dir = preprocessed_cache_dir
        self.reuse_preprocessed_data = reuse_preprocessed_data

        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.preprocess_metadata_fn = preprocess_metadata_fn

        self.data_column = data_column
        self.label_column = label_column
        self.preprocess_idx_column = preprocess_idx_column

        self.sampling_rate = sampling_rate
        self.length = int(length * sampling_rate)
        self.offset = int(offset * sampling_rate)
        self.pad_token = pad_token

        self.enable_inmemory_cache = enable_inmemory_cache
        self.data_cache = {}

        if not os.path.exists(os.path.join(self.root_dir, self.split)):
            raise ValueError(f"Data Path {os.path.join(self.root_dir, self.split)} does not exist.")

        # reuse preprocessed data if it exists
        if self.reuse_preprocessed_data:
            if self.preprocessed_cache_dir is not None:
                if not os.path.exists(self.preprocessed_cache_dir):
                    os.makedirs(self.preprocessed_cache_dir, exist_ok=True)

                preprocessed_meta_path = os.path.join(self.preprocessed_cache_dir, "preprocessed_meta_data.csv")
                if os.path.exists(preprocessed_meta_path):
                    self.meta_data = pd.read_csv(preprocessed_meta_path)

                    if self.data_column in self.meta_data.columns:
                        self.data_list = self.meta_data[self.data_column].tolist()
                    else:
                        raise ValueError(f"Meta file must have a {self.data_column} column containing the audio file paths.")
                    if self.label_column in self.meta_data.columns:
                        self.labels = self.meta_data[self.label_column].tolist()
                    else:
                        raise ValueError(f"Meta file must have a {self.label_column} column containing the labels.")

                    self.meta_data['preprocessed_location'] = self.set_preprocessed_locations(self.meta_data)
                else:
                    raise ValueError(f"Preprocessed meta data file {preprocessed_meta_path} does not exist, but reuse_preprocessed_data is True.")
        # recreate the meta data from the meta file, and init dynamic processing from scratch.
        elif meta_file is not None and not self.reuse_preprocessed_data:
            self.meta_data = pd.read_csv(meta_file)

            # alter the metadata if a preprocessing function is provided
            if self.preprocess_metadata_fn is not None and self.preprocess_fn is not None:
                self.meta_data = self.preprocess_metadata_fn(self.meta_data, self.preprocess_idx_column)
                if self.preprocessed_cache_dir is not None and os.path.exists(self.preprocessed_cache_dir):
                    ## comment out this line to avoid overwriting the preprocessed meta data file.
                    self.meta_data.to_csv(os.path.join(self.preprocessed_cache_dir, "preprocessed_meta_data.csv"), index=False)
                    self.meta_data['preprocessed_location'] = [None] * len(self.meta_data)

            if self.data_column in self.meta_data.columns:
                self.data_list = self.meta_data[self.data_column].tolist()
            else:
                raise ValueError(f"Meta file must have a {self.data_column} column containing the audio file paths.")
            if self.label_column in self.meta_data.columns:
                self.labels = self.meta_data[self.label_column].tolist()
            else:
                raise ValueError(f"Meta file must have a {self.label_column} column containing the labels.")
        # just use the data in the root_dir/split directory, no meta data, no preprocessing.
        else:
            # if no meta file is provided, neither reuse_preprocessed_data, we assume the data is in the root_dir/split directory, and to be loaded in the os listdir order.
            self.data_list = os.listdir(
                os.path.join(
                    self.root_dir,
                    self.split,
                    )
                )
            self.labels = None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if (not self.enable_inmemory_cache) or (idx not in self.data_cache):
            y = self.labels[idx] if self.labels is not None else torch.empty(0)
            audio_path = os.path.join(
                self.root_dir,
                self.split,
                self.data_list[idx]
                )
            x = self.load_audio(audio_path)

            # apply the preprocessing function to the audio data and labels if we're doing "dynamic" preprocessing
            if self.preprocess_fn is not None:
                x, y = self.preprocess(x, y, idx)

            # make sure the length is correct if length is passed
            if self.length is not None:
                if self.offset is None:
                    self.offset = 0
                x = self.trim_and_pad_audio(x)

            if self.enable_inmemory_cache:
                self.data_cache[idx] = (x, y)
        else:
            x, y = self.data_cache[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def set_preprocessed_locations(self, meta_data):
        """
        If a sample is also preprocessed and the file exists, simply point to it, otherwise set to None.
        The ones which are set to None will be preprocessed on the fly.
        """
        if 'preprocessed_location' not in meta_data.columns:
            meta_data['preprocessed_location'] = [None] * len(meta_data)
        preprocessed_location = meta_data['preprocessed_location'].to_list()
        for idx, location in enumerate(preprocessed_location):
            if location is None:
                preprocessed_file = os.path.join(self.preprocessed_cache_dir, f"{self.data_list[idx]}_preprocessed.pt")
                if os.path.exists(preprocessed_file):
                    preprocessed_location[idx] = preprocessed_file
        return preprocessed_location

    def load_audio(self, audio_path):
        if audio_path.endswith(".wav") or audio_path.endswith(".mp3"):
            x, _ = librosa.load(audio_path, sr=self.sampling_rate)
            # Notes: 1. if sr is None here, it will use the sr of the file.
            # 2. load has mono=True by default which averages samples across channels.
        elif audio_path.endswith(".csv"):
            x = np.loadtxt(audio_path)
            # since it's a csv file, we have no way of knowing the sampling rate and we assume it's the right one.
        elif audio_path.endswith(".npy"):
            x = np.load(audio_path)
        else:
            raise ValueError(f"Unsupported file type: {audio_path}")
        return x

    def trim_and_pad_audio(self, x):
        if len(x) < (self.length+self.offset):
            # use a pad_token to pad the audio if given by the user, else just repeat the audio.
            if self.pad_token is None:
                # Edge effects will be introduced but meaningful sounds aren't reversible in time,
                # i.e., cannot be flipped and concatenated to avoid edge effect.
                # Also, looping/repeating the sound might be useful for rythmic sounds like bird calls and music.
                x = np.tile(x, int(np.ceil((self.length+self.offset) / len(x))))
            else:
                pad_width = (self.length+self.offset) - len(x)
                x = np.pad(
                    x,
                    (0, pad_width),
                    mode="constant",
                    constant_values=self.pad_token
                    )
        x = x[self.offset:self.offset+self.length]
        return x

    def preprocess(self, x, y, idx):
        if self.preprocessed_cache_dir is not None:
            if self.meta_data['preprocessed_location'][idx] is None:
                with isolate_rng(idx):
                    x, y = self.preprocess_fn(x, y)
                # Set preprocessed location for all samples with same data_column value
                current_data = self.data_list[idx]
                same_data_mask = self.meta_data[self.data_column] == current_data
                self.meta_data.loc[same_data_mask, 'preprocessed_location'] = os.path.join(self.preprocessed_cache_dir, f"{self.data_list[idx]}_preprocessed.pt")
                # create directory for a species/ebird code which will be part of data_list (horlar/X124_preprocessed.pt)
                os.makedirs(os.path.dirname(self.meta_data['preprocessed_location'][idx]), exist_ok=True)
                torch.save({'x': x, 'y': y}, self.meta_data['preprocessed_location'][idx])
            else:
                cache = torch.load(self.meta_data['preprocessed_location'][idx], weights_only=True)
                x, y = cache['x'], cache['y']
        else:
            with isolate_rng(idx):
                x, y = self.preprocess_fn(x, y)

        # if we're doing sliding window, this gives all the windows as tuple with 2 lists.
        # We can get the correct window of this particular sample/idx by using the preprocess_index column.
        if self.preprocess_idx_column in self.meta_data.columns:
            # in the global meta data, we can just use global index and it will resolve to local preprocess index.
            current_idx = self.meta_data[self.preprocess_idx_column][idx]
            x, y = x[current_idx], y[current_idx]

        return x, y
