import torch
import copy


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

