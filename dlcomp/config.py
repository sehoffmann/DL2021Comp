import torch
import inspect

from dlcomp.models.autoencoder import SimpleAutoencoder


def remove_name(cfg):
    cfg2 = dict(cfg)
    del cfg2['name']
    return cfg2


def create_optimizer(name, params, **kwargs):
    torch_optimizers = dict(inspect.getmembers(torch.optim, inspect.isclass))

    if name in torch_optimizers:
        return torch_optimizers[name](params, **kwargs)
    else:
        raise ValueError(f'unknown optimizer {name}')        


def optimizer_from_config(cfg, params):
    kwargs = remove_name(cfg)
    return create_optimizer(cfg['name'], params, **kwargs)


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
    else:
        raise ValueError(f'unknown experiment {experiment}')