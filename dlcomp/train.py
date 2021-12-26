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



flags.DEFINE_string('default_config', 'default-cfg.yaml', 'path to the default configuration file')
flags.DEFINE_string('config', 'experiments-cfg/baseline.yaml', 'experiment specific configuration file')


def main(argv):

    with open(flags.FLAGS.default_config, 'r') as f:
        config = yaml.load(f, Loader)

    with open(flags.FLAGS.config, 'r') as f:
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
        group = None
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
    experiment.train()


if __name__=="__main__":
    app.run(main)