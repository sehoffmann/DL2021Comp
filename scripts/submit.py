import wandb
import torch
import numpy as np
import random
import os
from absl import flags, app

from kaggle.api.kaggle_api_extended import KaggleApi

from dlcomp.config import experiment_from_config, augmentation_from_config
from dlcomp.eval import infer_and_safe, infer_and_safe_ensemble
from dlcomp.util import set_seed


FLAGS = flags.FLAGS

flags.DEFINE_multi_string('runs', None, 'the id of the run to submit (or a list for ensembling)', required=True)
flags.DEFINE_bool('infer', True, 'whether to rerun the inference')
flags.DEFINE_integer('aug_iters', 1, 'if larger than 1, use an augmentation ensemble with that number of iterations')
flags.DEFINE_float('aug_strength', 1, 'if using augmentation ensemble, the factor to multiply the augmentation strength by')
flags.DEFINE_bool('save_images', False, 'whether to save inference images')
flags.DEFINE_string('checkpoint', None, 'the checkpoint to use')


def infer_normal(experiment, outdir, device):
    return infer_and_safe(
        outdir, 
        experiment.test_dl, 
        experiment.ema_model, 
        device, 
        save_images=FLAGS.save_images
    )


def infer_aug_ensemble(experiment, outdir, device):
    aug_cfg = experiment.cfg['augmentation'].copy()
    if 'strength' in aug_cfg:
        strength = aug_cfg['strength']
        print(f'using augmentation strength {strength*FLAGS.aug_strength} instead of {strength}')
        aug_cfg['strength'] *= FLAGS.aug_strength

    #if 'proportion' in aug_cfg:
    #    aug_cfg['proportion'] = 1.0

    augmentation = augmentation_from_config(aug_cfg)

    return infer_and_safe_ensemble(
        outdir, 
        experiment.test_dl, 
        augmentation, 
        experiment.ema_model, 
        device, 
        iterations=FLAGS.aug_iters,
        save_images=FLAGS.save_images
    )


def get_predictions(api, run_id, run_dir):
    run = api.run(f'sehoffmann/dlcomp/runs/{run_id}')
    config = dict(run.config)

    # setup device
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    # setup randomness
    set_seed(config['seed'])

    # run inference or load cvv results
    experiment = experiment_from_config(config)
    if FLAGS.infer:
        checkpoint = 'models/' + FLAGS.checkpoint if FLAGS.checkpoint else 'models/best.pth'
        best_model = run.file(checkpoint).download(run_dir)  # io.TextIOWrapper
        experiment.restore(best_model.name)
        
        if FLAGS.aug_iters > 1:
            csv_path = infer_aug_ensemble(experiment, run_dir, config['device'])
        else:
            csv_path = infer_normal(experiment, run_dir, config['device'])
    else:
        csv_path = run.file('kaggle_prediction.csv').download(run_dir).name

    preds = np.loadtxt(csv_path, delimiter=',')[:, 1].reshape(-1, 3, 96, 96)
    return preds


def build_msg(api):
    if len(FLAGS.runs) == 1:
        run_id = FLAGS.runs[0]
        run = api.run(f'sehoffmann/dlcomp/runs/{run_id}')

        test_loss = run.summary['best/test/ema_loss']
        msg = f'{run.name} https://wandb.ai/sehoffmann/dlcomp/runs/{run_id}'
        
        if FLAGS.checkpoint:
            msg += f' {FLAGS.checkpoint}'
        
        msg += f' (test-loss: {test_loss:.5f})'
    
    else:
        msg = f'[ensemble: ' + '  |  '.join(FLAGS.runs) + ']'

    if FLAGS.aug_iters > 1:
        msg = f'[augmentation ensemble {FLAGS.aug_iters} augstr: {FLAGS.aug_strength:.2f}] ' + msg
    
    return msg


def main(vargs):
    api = wandb.Api()

    path = f'results/ensemble/{random.randint(1, 1e12):x}'
    os.makedirs(path)

    predictions = []
    for run_id in FLAGS.runs:
        run_path = path + f'/{run_id}'
        os.mkdir(run_path)
        predictions += [get_predictions(api, run_id, run_path)]

    predictions = np.mean(predictions, axis=0)

    # save as csv
    flattened = predictions.flatten()[:, None]
    indices = np.arange(len(flattened))[:, None]
    csv_data = np.concatenate([indices, flattened], axis=1)
    csv_path = path + '/predictions.csv'
    np.savetxt(csv_path, csv_data, delimiter=",", header='Id,Value', fmt='%d,%f')

    # submit to kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    msg = build_msg(api)
    #kaggle_api.competition_submit(csv_path, msg, 'uni-tuebingen-deep-learning-2021')


if __name__ == '__main__':
    app.run(main)