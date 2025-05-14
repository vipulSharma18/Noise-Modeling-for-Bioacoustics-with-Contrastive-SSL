from ssl_bioacoustics.utils import isolate_rng
from torch.utils.data import Dataset


class FrozenNoiseDataset(Dataset):
    def __init__(self, dataset, noise_transform=None, transform=None, **kwargs):
        self.dataset = dataset(**kwargs)
        self.noise_transform = noise_transform
        self.transform = transform
        self.data_cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.data_cache:
            x, y = self.dataset[idx]
            if self.noise_transform is not None:
                with isolate_rng(idx):
                    x = self.noise_transform(x)
            self.data_cache[idx] = (x, y)
        else:
            x, y = self.data_cache[idx]

        if len(self.data_cache) == len(self.dataset):
            del self.dataset  # free memory cause dataset will also have its own cache

        if self.transform is not None:
            x = self.transform(x)

        return x, y
