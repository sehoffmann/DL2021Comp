import wandb
import numpy as np
import os
import random
import torch
from absl import flags, app
from torchvision.utils import save_image

from kaggle.api.kaggle_api_extended import KaggleApi

from dlcomp.config import experiment_from_config
from dlcomp.eval import infer_and_safe
from dlcomp.util import set_seed


FLAGS = flags.FLAGS

flags.DEFINE_list('runs', None, 'comma-separated list of run ids', required=True)
flags.DEFINE_bool('saveimgs', True, 'save ensemble images')

def main(vargs):
    api = wandb.Api()
    
    path = f'results/ensemble/{random.randint(1, 1e12):x}'
    os.makedirs(path)
    print(f'writing to {path}')

    run_predictions = []
    for run_id in FLAGS.runs:
        print(f'downloading {run_id}')
        run = api.run(f'sehoffmann/dlcomp/runs/{run_id}')
        csv_file = run.file('kaggle_prediction.csv').download(path, replace=True)

        preds = np.loadtxt(csv_file, delimiter=',')[:, 1].reshape(-1, 3, 96, 96)
        run_predictions.append(preds)

    run_predictions = np.stack(run_predictions, axis=0)
    ensemble_predictions = np.mean(run_predictions, axis=0)
    for i, img in enumerate(ensemble_predictions):
        save_image(torch.tensor(img / 255), path + f'/img{i:03d}.png')


    # save as csv
    flattened = ensemble_predictions.flatten()[:, None]
    indices = np.arange(len(flattened))[:, None]
    csv_data = np.concatenate([indices, flattened], axis=1)
    csv_path = path + '/ensemble.csv'
    np.savetxt(csv_path, csv_data, delimiter=",", header='Id,Value', fmt='%d,%f')

    # submit to kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()


    msg = f'ensemble: ' + '  |  '.join(FLAGS.runs)
    kaggle_api.competition_submit(csv_path, msg, 'uni-tuebingen-deep-learning-2021')


if __name__ == '__main__':
    app.run(main)