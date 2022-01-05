import torch
import wandb
import logging
import pprint
from absl import flags, app

from dlcomp.config import experiment_from_config
from dlcomp.util import cleanup_wandb_config, set_seed

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


FLAGS = flags.FLAGS

flags.DEFINE_string('default_config', 'default-cfg.yaml', 'path to the default configuration file')
flags.DEFINE_string('config', 'experiments-cfg/baseline_0.yaml', 'experiment specific configuration file')
flags.DEFINE_bool('profile', False, 'activate profiler')


def main(argv):

    with open(FLAGS.default_config, 'r') as f:
        config = yaml.load(f, Loader)

    with open(FLAGS.config, 'r') as f:
        exp_config = yaml.load(f, Loader)
        if exp_config:
            config.update(exp_config)
        else:
            logging.warning('empty experiment config')
        
    wandb.init(
        project='dlcomp', 
        entity='sehoffmann',
        config=config,
        tags = [], 
        group = 'mean_teacher'
    )

    # retrieve config from wandb, cleanup dotted named, and write it back
    # important for sweeps
    config = dict(wandb.config)
    config = cleanup_wandb_config(config)

    print('='*80 + '\nConfig:\n' + '=' * 80)
    pprint.pprint(config)

    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 80)
    print(f'Using "{config["device"]}" device')
    print('=' * 80)

    # setup randomness
    set_seed(config['seed'])
    if config['deterministic_cudnn']:
        torch.backends.cudnn.deterministic = True

    # run experiment
    experiment = experiment_from_config(config)
    if FLAGS.profile:
        profile_experiment(experiment)
    else:
        experiment.train()


def profile_experiment(experiment):
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        experiment.train()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("profile_trace.json")


if __name__=="__main__":
    app.run(main)