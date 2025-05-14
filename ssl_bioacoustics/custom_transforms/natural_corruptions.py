"""
This module supports loading another environmental sound
or any dataset and using it as a noise.
"""

import torchaudio.transforms
import torch


class EnvironmentalNoise(torch.nn.Module):
    def __init__(self,
                 noise_source=None,
                 snr=10,
                 ):
        super(EnvironmentalNoise, self).__init__()
        self.noise_dataset = noise_source
        if len(self.noise_dataset) == 0:
            raise ValueError("Noise dataset is empty.")
        self.snr = snr
        self.transform = torchaudio.transforms.AddNoise()

    def forward(self, x):
        # sample from noise source dataset
        # hash x to create a deterministic index into the noise source dataset
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        idx = torch.randint(low=0, high=len(self.noise_dataset), size=(1,)).item()
        noise, _ = self.noise_dataset[idx]
        if not isinstance(noise, torch.Tensor):
            noise = torch.tensor(noise)

        # Keep on adding new samples till the noise is >= x.
        # If the class has very small num of samples, the noise can get repeated.
        while noise.shape[-1] < x.shape[-1]:
            idx = (idx + 1) % len(self.noise_dataset)
            new_noise, _ = self.noise_dataset[idx]
            if not isinstance(new_noise, torch.Tensor):
                new_noise = torch.tensor(new_noise)
            noise = torch.cat((noise, new_noise), dim=-1)

        noise = noise[..., :x.shape[-1]]
        noise = noise.view(x.size())
        noise = noise.to(x.device)

        snrs = torch.full(x.size()[:-1], self.snr)  # duplicate the snr for each channel as expected by torchaudio
        x_shape = x.shape[:-1]
        noise_shape = noise.shape[:-1]
        assert (x_shape) == (noise_shape), f"Non-last dimensions of x and noise must match. Got {x.shape}, {noise.shape}."
        x = self.transform.forward(x, noise, snrs)
        return x
