import cv2
import torch
import numpy as np

from imgaug.augmentables import KeypointsOnImage, Keypoint

from dlcomp.data_handling import loaders_from_config
from dlcomp.models.layers import AffineTransform
from dlcomp.util import affine_keypoints, affine_matrix_from_kps, cv_affine_to_torch, inverse_affine, norm_img_and_chw, unnorm_img_and_hwc, update_ema_model
from dlcomp.augmentations import to_tensor
from .default import DefaultLoop


class MeanTeacherLoop(DefaultLoop):

    def __init__(self, cfg):
        super().__init__(cfg)
        
        exp_cfg = cfg['experiment']
        self.trade_losses = exp_cfg.pop('trade_losses')
        self.max_trade = exp_cfg.pop('max_trade')
        self.consistency_loss_scale = exp_cfg.pop('consistency_loss_scale')
        self.ramp_up = exp_cfg.pop('ramp_up')

        self._mse_loss = torch.nn.MSELoss()
        self._affine_trans = AffineTransform(interpolation_mode='bicubic').to(self.device)
        self._consistency_loss = 0


    def setup_datasets(self, cfg):
        return loaders_from_config(cfg, transform=None)


    def step(self, X, Y):
        _,C,H,W = X.shape
        Y = Y.to(self.device, non_blocking=True)

        X = unnorm_img_and_hwc(X.numpy())

        kps = affine_keypoints((H,W,C))

        X1, kps_student = self.augmentation(images=X, keypoints=[kps] * X.shape[0])
        X1 = torch.Tensor(norm_img_and_chw(X1)).to(self.device, non_blocking=True)
        
        X2, kps_teacher = self.augmentation(images=X, keypoints=[kps] * X.shape[0])
        X2 = torch.Tensor(norm_img_and_chw(X2)).to(self.device, non_blocking=True)
        
        # calculate affine transformations
        affine_student = affine_matrix_from_kps(kps, kps_student, inverse=False)  # array of transformation matrices
        inv_affine_student = inverse_affine(affine_student)
        affine_student = torch.Tensor(cv_affine_to_torch(affine_student, H,W)).to(self.device, non_blocking=True)
        inv_affine_student = torch.Tensor(cv_affine_to_torch(inv_affine_student, H,W)).to(self.device, non_blocking=True)

        inv_affine_teacher = affine_matrix_from_kps(kps, kps_teacher, inverse=True)
        inv_affine_teacher = torch.Tensor(cv_affine_to_torch(inv_affine_teacher, H,W)).to(self.device, non_blocking=True)

        # inference
        pred_student = self.model(X1)
        pred_teacher = self.ema_model(X2)
        
        # logloss
        Y1 = self._affine_trans(Y, affine_student).detach()
        normal_loss = self.loss_fn(pred_student, Y1)  # torch.tensor !

        # consistency loss
        # first transform back to original position 
        pred_student_trans = self._affine_trans(pred_student, inv_affine_student)
        pred_teacher_trans = self._affine_trans(pred_teacher, inv_affine_teacher).detach()
        consistency_cost = self.consistency_loss_scale*self._mse_loss(255 * pred_student_trans, 255 * pred_teacher_trans)
        self._consistency_loss += consistency_cost

        w2 = self._consistency_weight()
        w2 *= self.max_trade if self.trade_losses else 1
        w1 = 1-w2 if self.trade_losses else 1
        loss = w1*normal_loss + w2*consistency_cost

        # step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        update_ema_model(self.model, self.ema_model, self.cfg['experiment']['ema_alpha'])

        return loss


    def validate_step(self, model, X,Y, is_test):
        _,C,H,W = X.shape
        Y = Y.to(self.device, non_blocking=True)

        if is_test:
            X = X.to(self.device)
            pred = model(X)
            return self.kaggle_loss(pred, Y)
        else:
            X_np = unnorm_img_and_hwc(X.numpy())

            kps_org = affine_keypoints((H,W,C))
            X_trans, kps_trans = self.augmentation(images=X_np, keypoints=[kps_org] * X.shape[0])
            X_trans = torch.Tensor(norm_img_and_chw(X_trans)).to(self.device, non_blocking=True)

            affine_matrices = affine_matrix_from_kps(kps_org, kps_trans, inverse=False)
            affine_matrices = torch.Tensor(cv_affine_to_torch(affine_matrices, H,W)).to(self.device, non_blocking=True)

            pred = model(X_trans)
            Y_trans = self._affine_trans(Y, affine_matrices)
            return self.loss_fn(pred, Y_trans)


    def _consistency_weight(self):
        t = min(1, self.batch / self.ramp_up)
        sigmoid = np.exp(-6*(1-t)**2.5)  # between 0 and 1
        return sigmoid


    def update_metrics(self, metrics):
        weight = self._consistency_weight()
        weight *= self.max_trade if self.trade_losses else 1
        
        metrics.update({
            'train/consistency_loss_weight': weight,
            'train/consistency_loss': self._consistency_loss / len(self.train_dl)
        })
        self._consistency_loss = 0
        return metrics

    
    @property
    def ema_alpha(self):
        ema_alpa =  super(MeanTeacherLoop, self).ema_alpha
        lower_bound = 0.8
        return lower_bound + (ema_alpa-lower_bound) * self._consistency_weight()