import torch
from torch import nn
import inspect

import dlcomp.augmentations as aug
from dlcomp.losses import KaggleLoss


def remove_name(cfg):
    cfg2 = dict(cfg)
    del cfg2['name']
    return cfg2


def optimizer_from_config(cfg, params):
    kwargs = remove_name(cfg)
    name = cfg['name']

    torch_optimizers = dict(inspect.getmembers(torch.optim, inspect.isclass))
    if name in torch_optimizers:
        return torch_optimizers[name](params, **kwargs)
    else:
        raise ValueError(f'unknown optimizer {name}')


def scheduler_from_config(cfg, optimizer):
    kwargs = remove_name(cfg)
    name = cfg['name']

    torch_optimizers = dict(inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass))
    if name in torch_optimizers:
        return torch_optimizers[name](optimizer, **kwargs)
    else:
        raise ValueError(f'unknown optimizer {name}')


def model_from_config(cfg):
    kwargs = remove_name(cfg)
    name = cfg['name']
    if cfg['name'] == 'SimpleAutoencoder':
        from dlcomp.models.autoencoder import SimpleAutoencoder
        return SimpleAutoencoder(**kwargs)
    elif name == 'Autoencoder':
        from dlcomp.models.autoencoder import Autoencoder
        return Autoencoder(**kwargs)
    elif name == 'DictionaryAutoencoder':
        from dlcomp.models.softdict import DictionaryAutoencoder
        return DictionaryAutoencoder(**kwargs)
    else:
        raise ValueError(f'unknown model {cfg["name"]}')


def experiment_from_config(cfg):
    experiment = cfg['experiment']['name']
    if experiment == 'default':
        from dlcomp.experiments.default import DefaultLoop
        return DefaultLoop(cfg)
    elif experiment == 'groundtruth':
        from dlcomp.experiments.groundtruth import Groundtruth
        return Groundtruth(cfg)
    elif experiment == 'mean_teacher':
        from dlcomp.experiments.mean_teacher import MeanTeacherLoop
        return MeanTeacherLoop(cfg)
    else:
        raise ValueError(f'unknown experiment {experiment}')


def augmentation_from_config(cfg):
    if not isinstance(cfg, str):
        name = cfg['name']
        kwargs = remove_name(cfg)
    else:
        name = cfg
        kwargs = {}

    return aug.augmentations[name](**kwargs)


def activation_from_config(cfg):
    if cfg is None:
        return nn.Identity()

    if not isinstance(cfg, str):
        name = cfg['name']
        kwargs = remove_name(cfg)
    else:
        name = cfg
        kwargs = {}

    if name == 'ReLU':
        return nn.ReLU(**kwargs)
    elif name == 'ELU':
        return nn.ELU(**kwargs)
    elif name == 'SELU':
        return nn.SELU(**kwargs)
    elif name == 'SiLU':  # swish
        return nn.SiLU(**kwargs)
    elif name == 'GELU':
        return nn.GELU(**kwargs)
    else:
        raise ValueError(f'unknown activation function {name}')


def loss_from_config(cfg):
    if not isinstance(cfg, str):
        name = cfg['name']
        kwargs = remove_name(cfg)
    else:
        name = cfg
        kwargs = {}

    if name == 'MSE':
        return nn.MSELoss(**kwargs)
    elif name == 'Kaggle':
        return KaggleLoss(**kwargs)
    else:
        raise ValueError(f'unknown loss function {name}')