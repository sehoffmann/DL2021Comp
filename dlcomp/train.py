import torch
import numpy as np
import wandb
import logging
import copy
from absl import flags, app

from dlcomp.config import experiment_from_config

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


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
        group = 'autoencoder'
    )

    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    # setup randomness
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
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