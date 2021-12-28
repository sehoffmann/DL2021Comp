import numpy as np
import torch
import os
import time
from torchvision.utils import save_image


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
                pic_num = int(i*dataloader.batch_size)
                for j, (train_img,test_img) in enumerate(zip(X,pred_X)):
                    save_image(train_img,f"{path}/{j+pic_num}_train.png")
                    save_image(test_img,f"{path}/{j+pic_num}_predicted.png")

    predictions =  np.concatenate(predictions)
    predictions *= 255
    predictions = np.expand_dims(predictions, 1)
    
    indices = np.expand_dims(np.arange(len(predictions)), 1)
    csv_data = np.concatenate([indices, predictions], axis=1)
    csv_file = path + ".csv"
    np.savetxt(csv_file, csv_data, delimiter=",", header='Id,Value', fmt='%d,%f')
    return csv_file