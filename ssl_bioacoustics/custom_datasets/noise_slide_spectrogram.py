import os
import time
import random
from torch.utils.data import Dataset
import torch
import torchvision
from ssl_bioacoustics.custom_transforms import (
    Spectrogram,
    SlidingWindowTransform,
)
from ssl_bioacoustics.custom_samplers import TransformSlideBySlide
from ssl_bioacoustics.utils import isolate_rng


class NoiseSlideSpectrogramGPU(Dataset):
    def __init__(
        self,
        nested_dataset,
        noise_transform=None,
        transform=None,
        enable_cache=True,
        spectrogram_gpu=False,
        random_noise=False,
        fixed_noise=False,
        cache_dir=None,
        window_size=1.0,
        hop_length=0.5,
        ):
        self.nested_dataset = nested_dataset
        self.noise_transform = noise_transform
        self.transform = transform
        self.enable_cache = enable_cache
        self.data_cache = []
        self.cache_dir = os.path.join(cache_dir, str(int(time.time())))
        self.cache_dir = os.path.join(self.cache_dir, os.getenv("SLURM_JOB_ID", str(random.randint(0, 1000))))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.random_noise = random_noise
        self.fixed_noise = fixed_noise
        self.spectrogram = Spectrogram(
            sampling_rate = 32000,
            spectrogram = "power_mel",
            convert_to_db = True,
            representation_mode = "rgb",
            compute_on_gpu_if_available=spectrogram_gpu,
            kwargs = {
                "n_mels": 128,
                "n_fft": 2048,
                "hop_length": 512
            }
        )
        self.sliding_window = SlidingWindowTransform(
            sampling_rate = 32000,
            window_size = window_size,
            hop_length = hop_length,
        )
        self.spectrogram_slide_by_slide = TransformSlideBySlide(
            slide_transforms = torchvision.transforms.Compose([self.spectrogram, torchvision.transforms.v2.ToImage()])
        )

    def __len__(self):
        return len(self.nested_dataset)

    def __getitem__(self, idx):
        if self.enable_cache and idx in self.data_cache:
            cache = torch.load(os.path.join(self.cache_dir, f"{idx}.pt"))
            x, y = cache['x'], cache['y']
            del cache
        else:
            x, y = self.nested_dataset[idx]
            # do the noise transform
            if self.noise_transform is not None:
                if self.random_noise:
                    x = self.noise_transform(x)
                if self.fixed_noise:
                    with isolate_rng(idx):
                        x = self.noise_transform(x)
            # do the sliding window transform
            x = self.sliding_window(x)
            # do the spectrogram transform slide-by-slide
            x = self.spectrogram_slide_by_slide(x)
            # when we want random noise, we won't be caching the data across epochs
            if self.enable_cache and not self.random_noise:
                if isinstance(x, torch.Tensor):
                    x = x.cpu()
                if isinstance(y, torch.Tensor):
                    y = y.cpu()
                torch.save({"x": x, "y": y}, os.path.join(self.cache_dir, f"{idx}.pt"))
                # self.data_cache.append(idx)
                # remove contention in data cache by reading from the folder structure after a write.
                self.data_cache = [int(x.split(".")[0]) for x in os.listdir(self.cache_dir)]

        # any other transform like random crop, random flip, etc.
        if self.transform is not None:
            x = self.transform(x)

        return x, y
