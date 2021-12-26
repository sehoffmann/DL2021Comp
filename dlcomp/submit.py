import wandb
import torch
from absl import flags, app

from kaggle.api.kaggle_api_extended import KaggleApi

from dlcomp.config import experiment_from_config
from dlcomp.eval import infer_and_safe


FLAGS = flags.FLAGS

flags.DEFINE_string('run', None, 'the id of the run to submit', required=True)

def main(vargs):
    api = wandb.Api()
    run = api.run(f'sehoffmann/dlcomp/runs/{FLAGS.run}')

    config = dict(run.config)
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    experiment = experiment_from_config(config)
    best_model = run.file('models/best.pth').download(experiment.model_dir)  # io.TextIOWrapper
    experiment.restore(best_model.name)

    csv_path = infer_and_safe(experiment.model_dir, experiment.test_dl, experiment.ema_model, config['device'])

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    msg = f'run: {FLAGS.run}'
    kaggle_api.competition_submit(csv_path, msg, 'uni-tuebingen-deep-learning-2021')

if __name__ == '__main__':
    app.run(main)