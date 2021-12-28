import wandb
import torch
from absl import flags, app

from kaggle.api.kaggle_api_extended import KaggleApi

from dlcomp.config import experiment_from_config
from dlcomp.eval import infer_and_safe


FLAGS = flags.FLAGS

flags.DEFINE_string('run', None, 'the id of the run to submit', required=True)
flags.DEFINE_bool('infer', True, 'whether to rerun the inference')


def main(vargs):
    api = wandb.Api()
    run = api.run(f'sehoffmann/dlcomp/runs/{FLAGS.run}')

    config = dict(run.config)
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    experiment = experiment_from_config(config)
    if FLAGS.infer:
        best_model = run.file('models/best.pth').download(experiment.model_dir)  # io.TextIOWrapper
        experiment.restore(best_model.name)
        csv_path = infer_and_safe(experiment.model_dir, experiment.test_dl, experiment.ema_model, config['device'], save_images=False)
    else:
        csv_path = run.file('kaggle_prediction.csv').download(experiment.model_dir).name


    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    test_loss = run.summary['best/test/ema_loss']
    msg = f'{run.name} https://wandb.ai/sehoffmann/dlcomp/runs/{FLAGS.run} (test-loss: {test_loss:.5f})'
    kaggle_api.competition_submit(csv_path, msg, 'uni-tuebingen-deep-learning-2021')


if __name__ == '__main__':
    app.run(main)