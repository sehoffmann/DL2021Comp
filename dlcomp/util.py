import torch
import numpy as np
import imgaug
import copy
import wandb


def cleanup_wandb_config(config, update=True):
    """
    Currently sweeps don't properly populate nested dicts.
    This function removes dotted names from the config, e.g. 'augmentation.strength',
    and turns them into nested dicts: config['augmentation']['strength'] = ...
    """
    removed_keys = []
    for key in config.copy():
        if '.' not in key:
            continue

        value = config[key]
        removed_keys.append(key)
        del config[key]

        parts = key.split('.')
        dct = config
        for part in parts[:-1]:
            if part in dct:
                assert isinstance(dct[part], dict)
                dct = dct[part]
            else:
                dct[part] = {}
                dct = dct[part]
        
        dct[parts[-1]] = value

    # update config on backend
    if update:
        api = wandb.Api()
        run = api.run(wandb.run.path)
        run.config = config
        run.update()

    # write changes back to wandb.config for consistency
    if wandb.run:
        wandb.config = wandb.Config()  # because there is no __del__()
        for k,v in config.items():
            wandb.config[k] = v

    return config


def set_seed(seed):
    """
    makes sure that seed is set consistently for all libraries
    """
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    imgaug.seed(seed)


def update_ema_model(model, ema_model, alpha):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        p2.data = alpha*p2 + (1-alpha) * p1.data


class EarlyStopping:

    def __init__(self, min_delta, grace_period):
        self.min_delta = min_delta
        self.grace_period = grace_period

        self.best_model = None
        self.best_epoch = None
        self.best_loss = None

    
    def update(self, epoch, model, loss):
        if not self.best_loss or loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model)
            return True
        else:
            return False


    def stop(self, epoch):
        return epoch - self.best_epoch > self.grace_period


    @classmethod
    def from_config(cls, cfg):
        return cls(cfg['min_delta'], cfg['grace'])

