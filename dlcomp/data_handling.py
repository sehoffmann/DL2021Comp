import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
from imgaug.augmentables import HeatmapsOnImage
from imgaug.augmentables import KeypointsOnImage, Keypoint

from dlcomp.util import set_seed, affine_matrix_from_kps
import dlcomp.augmentations as aug


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index].astype(np.uint8)
        y = self.targets[index].astype(np.uint8)

        return x, y

    def __len__(self):
        return len(self.data)


class AugmentedDataset(Dataset):

    def __init__(self, source_ds, transform=None, to_tensor=True, return_affine=False):
        self.source_ds = source_ds
        self.transform = transform
        self.to_tensor = to_tensor
        self.return_affine = return_affine


    def __getitem__(self, index):
        x, y = self.source_ds[index]
        H,W,C = x.shape

        heatmap = HeatmapsOnImage(y.astype('f4'), shape=x.shape, min_value=0, max_value=255)
        kps = KeypointsOnImage([
            Keypoint(x=0, y=0),
            Keypoint(x=0, y=H-1),
            Keypoint(x=W-1, y=0),
        ], shape=(H,W,C))

        if self.transform:
            x, y, kps_trans = self.transform(image=x, heatmaps=heatmap, keypoints=kps)
            y = y.get_arr().astype(np.uint8)
            M = affine_matrix_from_kps(kps, kps_trans, inverse=True)
        else:
            M = np.eye(3)[:2]  # 2x3 matrix

        if self.to_tensor:
            x, y = aug.to_tensor(x), aug.to_tensor(y)
 
        if self.return_affine:
            return x, y, M
        else:
            return x, y


    def __len__(self):
        return len(self.source_ds)


def seed_worker(worker_id):
    torch_seed = torch.initial_seed()
    set_seed(torch_seed)  # make sure that worker seed is propagated to 3rd party libs


def load_test_dataset(path):
    data = np.load(path)
    return NumpyDataset(data, np.zeros_like(data))


def load_train_dataset(noisy_path, label_path):
    x_data = np.load(noisy_path)
    y_data = np.load(label_path)
    return NumpyDataset(x_data, y_data)


def get_train_loaders(noisy_path, label_path, transform, to_tensor, return_affine, val_split, batch_size, shuffle, num_workers):
    ds_raw = load_train_dataset(noisy_path, label_path)
    
    N = len(ds_raw)
    train_set_raw, val_set_raw = torch.utils.data.dataset.random_split(
        ds_raw, 
        [int(N * (1-val_split)), int(N*val_split)]
    )

    train_set_aug = AugmentedDataset(train_set_raw, transform, to_tensor=to_tensor, return_affine=return_affine)
    val_set_aug = AugmentedDataset(val_set_raw, transform, to_tensor=to_tensor, return_affine=return_affine)
    val_set_raw = AugmentedDataset(val_set_raw, None, to_tensor=to_tensor)

    train_dl = DataLoader(
        train_set_aug, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )

    val_dl = DataLoader(
        val_set_aug, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )

    val_dl_raw = DataLoader(
        val_set_raw, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )

    return train_dl, val_dl, val_dl_raw


def get_test_loaders(path, transform, to_tensor, batch_size, shuffle, num_workers):
    ds = load_test_dataset(path)
    ds_aug = AugmentedDataset(ds, transform, to_tensor=to_tensor)
    dl = DataLoader(
        ds_aug, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )

    return dl


def loaders_from_config(cfg, transform, to_tensor=True, return_affine=False):
    train_dl, val_dl, val_dl_raw = get_train_loaders(
        cfg['train_noise_path'],
        cfg['train_clean_path'],
        transform,
        to_tensor,
        return_affine,
        cfg['validation_split'],
        cfg['batch_size'],
        True,
        cfg['io_threads']
    )

    test_dl = get_test_loaders(
        cfg['test_path'], 
        None, 
        to_tensor, 
        cfg['batch_size'], 
        False, 
        cfg['io_threads']
    )

    return train_dl, val_dl, val_dl_raw, test_dl
