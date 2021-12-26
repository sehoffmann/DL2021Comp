import torch
from torch.utils.data import DataLoader , Dataset

import numpy as np
from PIL import Image

import dlcomp.augmentations as aug

class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index].astype(np.uint8)

        y = Image.fromarray(self.targets[index].astype(np.uint8))
        y = aug.to_tensor(y)

        return x, y

    def __len__(self):
        return len(self.data)


class AugmentedDataset(Dataset):

    def __init__(self, source_ds, transform):
        self.source_ds = source_ds
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.source_ds[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.source_ds)


def load_test_dataset(path):
    data = np.load(path)
    return NumpyDataset(data, np.zeros_like(data))


def load_train_dataset(noisy_path, label_path):
    x_data = np.load(noisy_path)
    y_data = np.load(label_path)
    return NumpyDataset(x_data, y_data)


def get_train_loaders(noisy_path, label_path, transform, val_split, batch_size, shuffle, num_workers):
    ds_raw = load_train_dataset(noisy_path, label_path)
    
    N = len(ds_raw)
    train_set_raw, val_set_raw = torch.utils.data.dataset.random_split(
        ds_raw, 
        [int(N * (1-val_split)), int(N*val_split)]
    )

    train_set_aug = AugmentedDataset(train_set_raw, transform)
    val_set_aug = AugmentedDataset(val_set_raw, transform)
    val_set_raw = AugmentedDataset(val_set_raw, aug.to_tensor)

    train_dl = DataLoader(train_set_aug, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dl = DataLoader(val_set_aug, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dl_raw = DataLoader(val_set_raw, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_dl, val_dl, val_dl_raw


def get_test_loaders(path, transform, batch_size, shuffle, num_workers):
    ds = load_test_dataset(path)
    ds_aug = AugmentedDataset(ds, transform)
    return DataLoader(ds_aug, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def loaders_from_config(cfg, transform):
    train_dl, val_dl, val_dl_raw = get_train_loaders(
        cfg['train_noise_path'],
        cfg['train_clean_path'],
        transform,
        cfg['validation_split'],
        cfg['batch_size'],
        True,
        cfg['io_threads']
    )

    test_dl = get_test_loaders(cfg['test_path'], aug.to_tensor, cfg['batch_size'], False, cfg['io_threads'])

    return train_dl, val_dl, val_dl_raw, test_dl
