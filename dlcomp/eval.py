import numpy as np
import torch
import os
import time
from torchvision.utils import save_image
from dlcomp.util import affine_keypoints, affine_matrix_from_kps, cv_affine_to_torch, inverse_affine, norm_img_and_chw, unnorm_img_and_hwc, update_ema_model
from dlcomp.models.layers import AffineTransform


def _store_predictions(path, predictions):
    predictions =  np.concatenate(predictions)
    predictions *= 255
    predictions = np.expand_dims(predictions, 1)
    
    indices = np.expand_dims(np.arange(len(predictions)), 1)
    csv_data = np.concatenate([indices, predictions], axis=1)
    np.savetxt(path, csv_data, delimiter=",", header='Id,Value', fmt='%d,%f')


def _store_images(path, X, pred_X, offset):
    for j, (train_img,test_img) in enumerate(zip(X,pred_X)):
        save_image(train_img,f"{path}/{j+offset}_train.png")
        save_image(test_img,f"{path}/{j+offset}_predicted.png")


def infer_and_safe_ensemble(outdir, dataloader, augmentation, model, device, iterations=10, save_images=True):
    path = outdir + '/kaggle_prediction'
    if save_images:
        os.mkdir(path)

    affine_trans = AffineTransform(interpolation_mode='bicubic').to(device)

    predictions = []
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloader):
            _,C,H,W = X.shape
            
            ensemble = []
            for _ in range(iterations):
                X_np = unnorm_img_and_hwc(X.numpy())

                kps_org = affine_keypoints((H,W,C))
                X_trans, kps_trans = augmentation(images=X_np, keypoints=[kps_org] * X.shape[0])
                X_trans = torch.Tensor(norm_img_and_chw(X_trans)).to(device)

                affine_matrices = affine_matrix_from_kps(kps_org, kps_trans, inverse=True)
                affine_matrices = torch.Tensor(cv_affine_to_torch(affine_matrices, H,W)).to(device)

                pred = affine_trans(model(X_trans), affine_matrices)
                ensemble += [pred]


            X = X.to(device)
            pred_org = model(X)
            ensemble += [pred_org] * int(iterations * 0.5)

            ensemble_pred = torch.stack(ensemble, axis=1).mean(axis=1)
            predictions += [ensemble_pred.cpu().numpy().flatten()]

            if save_images:
                offset = int(i*dataloader.batch_size)
                _store_images(path, X, ensemble_pred, offset)
    
    _store_predictions(path + '.csv', predictions)
    return path + '.csv'


def infer_and_safe(outdir, dataloader, model, device, save_images=True):
    path = outdir + '/kaggle_prediction'
    if save_images:
        os.mkdir(path)

    predictions = []
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloader):
            X = X.to(device)
            pred_X = model(X)
            flat_pred_X = pred_X.cpu().numpy().flatten()
            predictions.append(flat_pred_X)

            if save_images:
                offset = int(i*dataloader.batch_size)
                _store_images(path, X, pred_X, offset)

    _store_predictions(path + '.csv', predictions)
    return path + '.csv'