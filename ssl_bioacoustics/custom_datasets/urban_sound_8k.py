"""
This script defines a custom dataset class for the UrbanSound8K dataset.
"""

import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
from PIL import Image
import os


class UrbanSoundDataset(Dataset):
    """
    classID             class
        0   air_conditioner
        1          car_horn
        2  children_playing
        3          dog_bark
        4          drilling
        5     engine_idling
        6          gun_shot
        7        jackhammer
        8             siren
        9      street_music
    """
    def __init__(self,
                 root_dir=None,
                 fold=0,
                 csv_file=None,
                 transform=None,
                 split="train",
                 sampling_rate=None,
                 classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.sampling_rate = sampling_rate
        if classes is None:
            self.classes = list(self.annotations['class'].unique())
        elif type(classes) is list:
            self.classes = [str(i) for i in classes]
        elif type(classes) is not list:
            self.classes = [str(classes)]
        else:
            raise ValueError("Classes must be a list, a string, or None.")
        # Filter the annotations for the current fold
        if split == "train":
            self.current_fold_annotations = self.annotations[
                (self.annotations['fold'] != fold) &
                (self.annotations['class'].isin(self.classes))
                ]
        else:
            self.current_fold_annotations = self.annotations[
                (self.annotations['fold'] == fold) &
                (self.annotations['class'].isin(self.classes))
                ]

    def __len__(self):
        return len(self.current_fold_annotations)

    def __getitem__(self, idx):
        """
        The item could be an image or a wav file.
        """
        slice_filename = self.current_fold_annotations.iloc[idx]['slice_file_name']
        fold = self.current_fold_annotations.iloc[idx]['fold']
        slice_path = os.path.join(self.root_dir, f'fold{fold}', slice_filename)

        if slice_filename.endswith('.wav'):
            if self.sampling_rate is None:
                self.sampling_rate = 22050  # default for librosa
            x, _ = librosa.load(slice_path, sr=self.sampling_rate)
        elif slice_filename.endswith(".csv"):
            x = np.loadtxt(slice_path)
        elif slice_filename.endswith('.png') or slice_filename.endswith('.jpg') or slice_filename.endswith('.jpeg'):
            x = Image.open(slice_path).convert('RGB')

        if self.transform is not None:
            x = self.transform(x)

        label = self.current_fold_annotations.iloc[idx]['classID']

        return x, label
