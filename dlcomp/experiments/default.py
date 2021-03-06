import torch
import wandb
import sys
import time
import os
import shutil
import numpy as np
from torchinfo import summary

import dlcomp.augmentations as aug
from dlcomp.data_handling import loaders_from_config

from dlcomp.config import augmentation_from_config, loss_from_config, model_from_config, optimizer_from_config, scheduler_from_config
from dlcomp.eval import infer_and_safe
from dlcomp.util import EarlyStopping, update_ema_model
from dlcomp.losses import KaggleLoss

class DefaultLoop:

    def __init__(self, cfg):
        self.cfg = cfg

        if wandb.run:
            self.model_dir = wandb.run.dir
        else:
            self.model_dir = cfg['out_path'] + f"/run_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
            os.makedirs(self.model_dir)

        exp_cfg = cfg['experiment']
        self.ema_alpha = exp_cfg.pop('ema_alpha')

        self.augmentation = augmentation_from_config(cfg['augmentation'])
        self.train_dl, self.val_dl, self.val_dl_raw, self.test_dl = self.setup_datasets(cfg)
        self.device = cfg['device']
        
        self.early_stopping = EarlyStopping.from_config(cfg['early_stopping'])

        self.model = model_from_config(cfg['model']).to(self.device)
        if cfg['data_parallel']:
            self.model = torch.nn.DataParallel(self.model)


        self.ema_model = model_from_config(cfg['model']).to(self.device).requires_grad_(False)
        self.optimizer = optimizer_from_config(cfg['optimizer'], self.model.parameters())
        self.scheduler = scheduler_from_config(cfg['scheduler'], self.optimizer) if 'scheduler' in cfg else None

        self.loss_fn = loss_from_config(cfg['loss']).to(self.device)
        self.kaggle_loss = KaggleLoss().to(self.device)

        if wandb.run:
            self.setup_wandb()

        self.summary()


    def summary(self):
        print('-' * 50)
        print(self.model)
        print('-' * 50)

        self.model.eval()
        X,Y = next(iter(self.train_dl))
        X,Y = self.prepare_batch(X,Y)
        model_statistics = summary(self.model, input_data=X, col_names=('kernel_size', 'output_size', 'num_params'))
        
        print('-' * 50)

        if wandb.run:
            wandb.run.summary['params'] = model_statistics.trainable_params
            wandb.run.summary['params_size'] = model_statistics.total_params*4 / 1e6
            wandb.run.summary['pass_size'] = model_statistics.total_output*4 / 1e6


    def setup_wandb(self):
        wandb.watch(self.model)

        wandb.define_metric('epoch', hidden=True)
        
        wandb.define_metric('train/loss', step_metric='epoch')
        wandb.define_metric('val/loss', step_metric='epoch')
        wandb.define_metric('val/ema_loss', step_metric='epoch')
        wandb.define_metric('test/loss', step_metric='epoch')
        wandb.define_metric('test/ema_loss', step_metric='epoch')

        wandb.define_metric('perf/epoch_time', summary='mean', step_metric='epoch', hidden=True)
        wandb.define_metric('perf/step_time', summary='mean', step_metric='epoch', hidden=True)
        wandb.define_metric('perf/throughput', summary='mean', step_metric='epoch', hidden=True)
        wandb.define_metric('perf/val_time', summary='mean', step_metric='epoch', hidden=True)

        wandb.define_metric('lr', step_metric='epoch', hidden=True)

        wandb.define_metric('test-images', step_metric='epoch', hidden=True)

        # set step to 1 instead of 0
        wandb.log(data={}, step=1)


    def setup_datasets(self, cfg):
        return loaders_from_config(cfg, self.augmentation)


    def train(self):
        self.epoch = 0
        self.batch = 0
        early_stop = False
        while self.epoch < self.cfg['epochs'] and not early_stop:
            self.epoch += 1

            print(f"Epoch {self.epoch}")
            print("-" * 50)
            
            ovh_t1 = time.perf_counter()

            # train
            t1 = time.perf_counter()
            train_loss = self.train_epoch()
            n_steps = len(self.train_dl)
            t2 = time.perf_counter()

            # validate
            val_loss = self.validate(self.model, self.val_dl)
            ema_val_loss = self.validate(self.ema_model, self.val_dl)

            test_loss = self.validate(self.model, self.val_dl_raw, is_test=True)
            ema_test_loss = self.validate(self.ema_model, self.val_dl_raw, is_test=True)

            ovh_t2 = time.perf_counter()

            metrics = {
                'train/loss': train_loss, 
                'val/loss': val_loss, 
                'val/ema_loss': ema_val_loss,
                'test/loss': test_loss,
                'test/ema_loss': ema_test_loss,

                'perf/epoch_time': t2-t1,
                'perf/step_time': 1000 * (t2-t1) / n_steps,  # in ms
                'perf/throughput': (n_steps * self.cfg['batch_size']) / (t2-t1),
                'perf/val_time': (ovh_t2 - ovh_t1) - (t2-t1)
            }

            new_best_model = self.early_stopping.update(self.epoch, self.ema_model, ema_test_loss) 
            if new_best_model:
                print('new best model!')
                for k,v in metrics.items():
                    wandb.run.summary['best/' + k] = v
                wandb.run.summary['best/epoch'] = self.epoch

            early_stop = self.early_stopping.stop(self.epoch) 
            if early_stop:
                print(f'no improvement since {self.early_stopping.grace_period} epochs. stopping early')


            if self.epoch % 20 == 0:
                self.log_test_images(self.model)
            
           
            metrics = self.update_metrics(metrics)
            print('\n'.join([f'{k}: {v:>7f}' for k,v in metrics.items()]))
            metrics.update({'lr': self.get_lr(), 'epoch': self.epoch, 'batch': self.batch})
            wandb.log(metrics)

            # Update LR
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

            self.save_models(is_best=new_best_model)

            print("-" * 50)

            sys.stdout.flush()
            sys.stderr.flush()

        infer_and_safe(self.model_dir, self.test_dl, self.early_stopping.best_model, self.device, save_images=False)
        print("Done!")


    def get_lr(self):
        if not self.scheduler or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pgroup = self.optimizer.param_groups[0]
            return pgroup['lr']
        else:
            return self.scheduler.get_last_lr() 


    def train_epoch(self):
        N_batches = len(self.train_dl)

        self.model.train()
        self.ema_model.train()

        epoch_loss = torch.tensor(0.0, device=self.device)
        for _, (X, Y) in enumerate(self.train_dl):
            self.batch += 1
            epoch_loss += self.step(X,Y)

        mean_loss = epoch_loss/N_batches
        return mean_loss.item()


    def step(self, X, Y):
        X, Y = self.prepare_batch(X,Y)

        pred = self.model(X)
        loss = self.loss_fn(pred, Y)  # torch.tensor !
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        update_ema_model(self.model, self.ema_model, self.ema_alpha)

        return loss


    def validate(self, model, dl, is_test=False):
        N_batches = len(self.val_dl)
        
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in dl:
                val_loss += self.validate_step(model, X,Y, is_test)

        return (val_loss / N_batches).item()


    def validate_step(self, model, X,Y, is_test):
        X, Y = self.prepare_batch(X,Y)
        pred = model(X)
        if is_test:
            return self.kaggle_loss(pred, Y)
        else:
            return self.loss_fn(pred, Y)


    def save_models(self, is_best=False):
        directory = self.model_dir + f'/models' 
        os.makedirs(directory, exist_ok=True)

        model_path = directory + f'/epoch{self.epoch}.pth'
        latest_path = directory + '/latest.pth'
        best_path = directory + '/best.pth'

        data =  {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }

        torch.save(data, latest_path)

        if self.epoch % self.cfg['save_every'] == 0:
            torch.save(data, model_path)
        
        if is_best:
            torch.save(data, best_path)

        wandb.save('models/*.pth')  # upload models as soon as possible


    def restore(self, model_path):
        checkpoint = torch.load(model_path)

        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        

    def log_test_images(self, model):
        predictions = []
        for i, (X, Y) in enumerate(self.test_dl):
            outputs = self.inference(model, X)
            predictions.append(outputs)

        predictions = np.concatenate(predictions)
        imgs = np.moveaxis(predictions, 1, -1)  # CHW -> HWC
        test_images = [wandb.Image(img) for img in imgs]
        wandb.log({'test-images': test_images, 'epoch': self.epoch}, commit=False)


    def inference(self, model, X):
        X, _ = self.prepare_batch(X, torch.Tensor([0.0]))
        return model(X).detach().cpu().numpy()

    
    def prepare_batch(self, X,Y):
        """
        Takes in the output produced by the dataset and produces
        torch.Tensor's ready for inference
        """
        return X.to(self.device), Y.to(self.device, non_blocking=True)


    def update_metrics(self, metrics):
        """
        hook point at end of epoch
        """
        return metrics