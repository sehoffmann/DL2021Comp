from .default import DefaultLoop
import numpy as np
import torch
from dlcomp.util import unnorm_img_and_hwc, norm_img_and_chw


class DomainAdaptationLoop(DefaultLoop):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.aug_threshold = cfg['domain_adaptation']['aug_threshold']
        self.dom_lambda = cfg['domain_adaptation']['dom_lambda']

        self.domain_classification_loss = torch.nn.BCELoss()

    def step(self, X, Y):
        # N, C, H, W = X.shape

        N, C, H, W = X.shape

        X = unnorm_img_and_hwc(X.numpy())

        Y = Y.to(self.device, non_blocking=True)

        # decide which samples should be augmented
        # shape N
        should_augment = np.random.rand(N)
        should_augment_multi = np.repeat(should_augment, C*H*W).reshape(X.shape)
        augs = self.augmentation(images=X) 

        X_comb = np.where(should_augment_multi > self.aug_threshold, X, augs)

        X_comb = torch.Tensor(norm_img_and_chw(X_comb)).to(self.device, non_blocking=True)

        # inference
        pred_label, pred_domain = self.model(X_comb)

        # loss (EQ. 1)
        ## i
        self._prediction_loss = self.loss_fn(pred_label, Y)

        ## ii
        domain_labels = np.round(should_augment)
        domain_labels = torch.tensor(domain_labels, dtype=torch.float).to(self.device, non_blocking=True)
        self._domain_loss = self.domain_classification_loss(pred_domain, domain_labels) 

        ## combination (gradient reversal layer) 
        loss = self._prediction_loss - self.dom_lambda * self._domain_loss

        # step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    def validate_step(self, model, X,Y, is_test):
        X, Y = self.prepare_batch(X,Y)
        pred, __ = model(X)
        if is_test:
            return self.kaggle_loss(pred, Y)
        else:
            return self.loss_fn(pred, Y)

    def inference(self, model, X):
        X, _ = self.prepare_batch(X, torch.Tensor([0.0]))
        pred, __ = model(X)
        return pred.detach().cpu().numpy()
    
    def update_metrics(self, metrics):
        metrics.update({
            'train/domain_loss': self._domain_loss,
            'train/prediction_loss': self._prediction_loss
        })