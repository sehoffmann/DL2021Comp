from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import torch
import numpy as np
import imgaug
import cv2
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
    Makes sure that seed is set consistently for all libraries
    """
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    imgaug.seed(seed)


def update_ema_model(model, ema_model, alpha):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        p2.data = alpha*p2 + (1-alpha) * p1.data


def norm_img_and_chw(img):
        img = np.transpose(img, (0,3,2,1))  # HWC -> CHW
        return img.astype(np.float32) / 255


def unnorm_img_and_hwc(img):
        img = np.transpose(img, (0,3,2,1))  # CHW -> HWC
        return (img*255).astype(np.uint8)  # [0,1] f4 -> [0,255] uint8


def affine_keypoints(shape):
    """
    Returns keypoints suitable to infer affine transformations
    """
    H,W,C = shape
    kps =  KeypointsOnImage([
        Keypoint(x=0, y=0),
        Keypoint(x=0, y=H-1),
        Keypoint(x=W-1, y=0),
    ], shape=(H,W,C))
    return kps


def inverse_affine(M):
    """
    Returns the matrix of the inverse affine mapping of M
    """
    M_inv = np.linalg.inv(M[..., :2])
    b = M[..., 2, None]  # ... x 2 x 1
    b_inv = M_inv @ -b
    return np.concatenate([M_inv, b_inv], axis=-1)


def affine_matrix_from_kps(kps1, kps2, inverse=False):
    """
    Returns affine transformation that maps keypoints kps1 to kps2.
    Keypoints must be a triangle.

    If ksp2 is a list of size N, a single Nx3x2 array is returned
    """
    X1 = kps1.to_xy_array().T

    if isinstance(kps2, KeypointsOnImage):
        kps2 = [kps2]

    matrices = []
    for keypoints in kps2:
        X2 = keypoints.to_xy_array().T
        if inverse:
            matrices.append(cv2.getAffineTransform(X2.T, X1.T))  # maps X2 -> X1
        else:
            matrices.append(cv2.getAffineTransform(X1.T, X2.T))  # maps X1 -> X2

    if len(matrices) == 1:
        return matrices[0]
    else:
        return np.stack(matrices, axis=0)


def cv_affine_to_torch(M, H,W):
    """
    Converts affine matrices from opencv to ones usable by pytorch
    """
    # This is what 4+ hours of trial and error looks like
    # dont ask me how, or why this works
    # none of the coordinate systems are documentated...

    A = np.empty(M.shape[:-2] + (2,2))
    A[..., 0,0] = M[..., 1,1]
    A[..., 0,1] = M[..., 1,0]
    A[..., 1,0] = M[..., 0,1]
    A[..., 1,1] = M[..., 0,0]
    
    A_inv = np.linalg.inv(A)
    
    b = np.empty(M.shape[:-2] + (2,1))
    b[...,0,0] =  A_inv[...,0,0] + A_inv[...,0,1] - 1
    b[...,0,0] -=  (M[...,1,2]*A_inv[...,0,0] + M[...,0,2]*A_inv[...,0,1]) * 2/W
    b[...,1,0] = A_inv[...,1,1] + A_inv[...,1,0] - 1
    b[...,1,0] -= (M[...,0,2]*A_inv[...,1,1] + M[...,1,2]*A_inv[...,1,0]) * 2/H
    
    return np.concatenate((A_inv, b), axis=-1)


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

