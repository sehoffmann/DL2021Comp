import torch
import wandb
import sys
import time
import os

import dlcomp.augmentations as aug
from dlcomp.data_handling import loaders_from_config

from dlcomp.config import model_from_config, optimizer_from_config
from dlcomp.eval import infer_and_safe
from dlcomp.util import update_ema_model


class DefaultLoop:

    def __init__(self, cfg):
        self.cfg = cfg

        self.model_dir = wandb.run.dir #cfg['out_path'] + f"/run_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        #os.makedirs(self.model_dir)

        self.train_dl, self.val_dl, self.test_dl = loaders_from_config(cfg, aug.baseline)
        self.device = cfg['device']

        self.model = model_from_config(cfg['model']).to(self.device)
        self.ema_model = model_from_config(cfg['model']).to(self.device).requires_grad_(False)
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
            val_loss = self.validate(self.model)
            ema_val_loss = self.validate(self.ema_model)

            self.save_models()

            print(f"train_loss: {train_loss:>7f}\nval_loss: {val_loss:>7f}\nema_val_loss: {ema_val_loss:>7f}")
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'ema_val_loss': ema_val_loss, 'epoch': self.epoch, 'batch': self.batch})
            print("-" * 50)

            sys.stdout.flush()
            sys.stderr.flush()

        infer_and_safe(self.model_dir, self.test_dl, self.model, self.device)
        print("Done!")
            

    def train_epoch(self):
        N_batches = len(self.train_dl)

        self.model.train()
        epoch_loss= 0.0
        for _, (X, Y) in enumerate(self.train_dl):
            self.batch += 1
            epoch_loss += self.step(X,Y)

        mean_loss = epoch_loss/N_batches
        return mean_loss


    def step(self, X, Y):
        X, Y = X.to(self.device), Y.to(self.device)

        pred = self.model(X)
        loss = self.loss_fn(pred*255, Y*255)  # to be consistent with the kaggle loss.
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        update_ema_model(self.model, self.ema_model, self.cfg['experiment']['ema_alpha'])

        wandb.log({'loss': loss, 'batch': self.batch, 'epoch': self.epoch})
        return loss


    def validate(self, model):
        N_batches = len(self.val_dl)
        
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in self.val_dl:
                X, Y = X.to(self.device), Y.to(self.device)
                pred = model(X)
                val_loss += self.loss_fn(pred*255, Y*255).item()
        
        return val_loss / N_batches
    

    def save_models(self):
        directory = self.model_dir + f'/models' 
        os.makedirs(directory, exist_ok=True)

        model_path = directory + f'/epoch{self.epoch}.pth'
        latest_path = directory + '/latest.pth'

        data =  {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }

        torch.save(data, model_path)
        torch.save(data, latest_path)
        wandb.save('models/*.pth')  # upload models as soon as possible