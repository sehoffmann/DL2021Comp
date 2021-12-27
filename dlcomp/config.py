import torch
import inspect

from dlcomp.models.autoencoder import SimpleAutoencoder
import dlcomp.augmentations as aug

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
    if cfg['name'] == 'SimpleAutoencoder':
        return SimpleAutoencoder(**kwargs)
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
    else:
        raise ValueError(f'unknown experiment {experiment}')


def augmentation_from_config(name):
    if name == 'baseline':
        return aug.baseline
    elif name == 'weak':
        return aug.weak
    else:
        raise ValueError(f'unknown augmentation {name}')