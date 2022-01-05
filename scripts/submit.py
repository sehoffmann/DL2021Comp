import wandb
import torch
from absl import flags, app

from kaggle.api.kaggle_api_extended import KaggleApi

from dlcomp.config import experiment_from_config, augmentation_from_config
from dlcomp.eval import infer_and_safe, infer_and_safe_ensemble
from dlcomp.util import set_seed


FLAGS = flags.FLAGS

flags.DEFINE_string('run', None, 'the id of the run to submit', required=True)
flags.DEFINE_bool('infer', True, 'whether to rerun the inference')
flags.DEFINE_integer('aug_iters', 1, 'if larger than 1, use an augmentation ensemble with that number of iterations')
flags.DEFINE_float('aug_strength', 1, 'if using augmentation ensemble, the factor to multiply the augmentation strength by')
flags.DEFINE_bool('save_images', False, 'whether to save inference images')


def infer_normal(experiment, device):
    return infer_and_safe(
        experiment.model_dir, 
        experiment.test_dl, 
        experiment.ema_model, 
        device, 
        save_images=FLAGS.save_images
    )


def infer_aug_ensemble(experiment, device):
    aug_cfg = experiment.cfg['augmentation'].copy()
    if 'strength' in aug_cfg:
        strength = aug_cfg['strength']
        print(f'using augmentation strength {strength*FLAGS.aug_strength} instead of {strength}')
        aug_cfg['strength'] *= FLAGS.aug_strength

    #if 'proportion' in aug_cfg:
    #    aug_cfg['proportion'] = 1.0

    augmentation = augmentation_from_config(aug_cfg)

    return infer_and_safe_ensemble(
        experiment.model_dir, 
        experiment.test_dl, 
        augmentation, 
        experiment.ema_model, 
        device, 
        iterations=FLAGS.aug_iters,
        save_images=FLAGS.save_images
    )


def main(vargs):
    api = wandb.Api()
    run = api.run(f'sehoffmann/dlcomp/runs/{FLAGS.run}')
    config = dict(run.config)

    # setup device
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    # setup randomness
    set_seed(config['seed'])

    # run inference or load cvv results
    experiment = experiment_from_config(config)
    aug_ensemble = FLAGS.aug_iters > 1
    if FLAGS.infer:
        best_model = run.file('models/best.pth').download(experiment.model_dir)  # io.TextIOWrapper
        experiment.restore(best_model.name)
        
        csv_path = infer_and_safe(experiment.model_dir, experiment.test_dl, experiment.ema_model, config['device'], save_images=False)
        if aug_ensemble:
            csv_path = infer_aug_ensemble(experiment, config['device'])
        else:
            csv_path = infer_normal(experiment, config['device'])
    else:
        csv_path = run.file('kaggle_prediction.csv').download(experiment.model_dir).name

    # submit to kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    test_loss = run.summary['best/test/ema_loss']
    msg = f'{run.name} https://wandb.ai/sehoffmann/dlcomp/runs/{FLAGS.run} (test-loss: {test_loss:.5f})'
    if aug_ensemble:
        msg = f'[augmentation ensemble {FLAGS.aug_iters} augstr: {FLAGS.aug_strength:.2f}] ' + msg
    kaggle_api.competition_submit(csv_path, msg, 'uni-tuebingen-deep-learning-2021')


if __name__ == '__main__':
    app.run(main)