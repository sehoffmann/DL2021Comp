import torch
from torch.utils.data import DataLoader , Dataset

import numpy as np
from PIL import Image

import dlcomp.augmentations as aug

class NumpyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x=self.data[index].astype(np.uint8)
        x= self.transform(x)

        y = Image.fromarray(self.targets[index].astype(np.uint8))
        y = aug.to_tensor(y)
        return x, y

    def __len__(self):
        return len(self.data)



def load_test_dataset(path, transform):
    data = np.load(path)
    return NumpyDataset(data, np.zeros_like(data), transform=transform)


def load_train_dataset(noisy_path, label_path, transform):
    x_data = np.load(noisy_path)
    y_data = np.load(label_path)
    return NumpyDataset(x_data, y_data, transform=transform)


def get_train_loaders(noisy_path, label_path, transform, val_split, batch_size, shuffle, num_workers):
    ds = load_train_dataset(noisy_path, label_path, transform)
    N = len(ds)

    train_set, eval_set = torch.utils.data.dataset.random_split(
        ds, 
        [int(N * (1-val_split)), int(N*val_split)]
    )

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    eval_dl = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, eval_dl


def get_test_loaders(path, transform, batch_size, shuffle, num_workers):
    ds = load_test_dataset(path, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def loaders_from_config(cfg, transform):
    train_dl, eval_dl = get_train_loaders(
        cfg['train_noise_path'],
        cfg['train_clean_path'],
        transform,
        cfg['validation_split'],
        cfg['batch_size'],
        True,
        cfg['io_threads']
    )

    test_dl = get_test_loaders(cfg['test_path'], aug.to_tensor, cfg['batch_size'], False, cfg['io_threads'])

    return train_dl, eval_dl, test_dl
