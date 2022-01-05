from .default import DefaultLoop
import numpy as np
import torch
from dlcomp.util import unnorm_img_and_hwc, norm_img_and_chw


class DomainAdaptationLoop(DefaultLoop):

    def step(self, X, Y):
        # N, C, H, W = X.shape
        aug_threshold = 0.5
        lamb = 0.5

        N, C, H, W = X.shape

        X = unnorm_img_and_hwc(X.numpy())

        Y = Y.to(self.device, non_blocking=True)

        # decide which samples should be augmented
        # shape N
        should_augment = np.random.rand(N)
        should_augment_multi = np.repeat(should_augment, C*H*W).reshape(X.shape)
        augs = self.augmentation(images=X) 

        X_comb = np.where(should_augment_multi > aug_threshold, X, augs)

        X_comb = torch.Tensor(norm_img_and_chw(X_comb)).to(self.device, non_blocking=True)

        # inference
        pred_label, pred_domain = self.model(X_comb)

        # loss (EQ. 1)
        ## i
        prediction_loss = self.loss_fn(pred_label, Y)

        ## ii
        domain_labels = np.round(should_augment)
        domain_labels = torch.tensor(domain_labels, dtype=torch.float).to(self.device, non_blocking=True)
        domain_loss = self.loss_fn(pred_domain, domain_labels) 

        ## combination (gradient reversal layer) 
        loss = prediction_loss - lamb * domain_loss

        # step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
