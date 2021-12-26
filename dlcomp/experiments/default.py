import torch
import numpy as np
from torch.utils.data import dataloader
import wandb
import sys

import dlcomp.augmentations as aug
from dlcomp.data_handling import loaders_from_config

from dlcomp.config import model_from_config, optimizer_from_config
from dlcomp.eval import infer_and_safe


class DefaultLoop:

    def __init__(self, cfg):
        self.cfg = cfg

        self.train_dl, self.val_dl, self.test_dl = loaders_from_config(cfg, aug.baseline)
        self.device = cfg['device']

        self.model = model_from_config(cfg['model']).to(self.device)
        self.optimizer = optimizer_from_config(cfg['optimizer'], self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()

        print(self.model)


    def train(self):
        wandb.watch(self.model)

        self.epoch = 0
        self.batch = 0
        while self.epoch < self.cfg['epochs']:
            self.epoch += 1

            print(f"Epoch {self.epoch}")
            print("-" * 50)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"train_loss: {train_loss:>7f}\nval_loss: {val_loss:>7f}")
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': self.epoch, 'batch': self.batch})
            print("-" * 50)

            sys.stdout.flush()
            sys.stderr.flush()

        infer_and_safe(self.cfg['out_path'], self.test_dl, self.model, self.device)
        print("Done!")
            

    def train_epoch(self):
        N_batches = len(self.train_dl)

        self.model.train()
        epoch_loss= 0.0
        for _, (X, Y) in enumerate(self.train_dl):
            self.batch += 1
            X, Y = X.to(self.device), Y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred*255, Y*255)  # to be consistent with the kaggle loss.
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss
            wandb.log({'loss': loss, 'batch': self.batch, 'epoch': self.epoch})

        mean_loss = epoch_loss/N_batches
        return mean_loss


    def validate(self):
        N_batches = len(self.val_dl)
        
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in self.val_dl:
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred*255, Y*255).item()
        
        return val_loss / N_batches
    

    