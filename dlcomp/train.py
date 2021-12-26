import torch
import numpy as np
import wandb
import logging
import copy
from absl import flags, app

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlcomp.experiments.default import DefaultLoop

flags.DEFINE_string('default_config', 'config-defaults.yaml', 'path to the default configuration file')
flags.DEFINE_string('config', 'experiments-cfg/baseline.yaml', 'experiment specific configuration file')


def main(argv):

    with open(flags.FLAGS.config, 'r') as f:
        exp_config = yaml.load(f, Loader)
    
    wandb.init(
        project='my-test-project', 
        entity='sehoffmann',
        config=flags.FLAGS.default_config
    )

    if exp_config:
        wandb.config.update(exp_config)
    else:
        logging.warning('empty experiment config')

    config = copy.deepcopy(dict(wandb.config))

    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {config['device']} device")

    # setup randomness
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['deterministic_cudnn']:
        torch.backends.cudnn.deterministic = True

    # run experiment
    experiment = config['experiment']['name']
    if experiment == 'default':
        DefaultLoop(config).train()
    else:
        logging.critical(f'unknown experiment {experiment}')
        return 1




# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")
#
# model = NeuralNetwork(width, hidden_layers)
# model.load_state_dict(torch.load("model.pth"))

if __name__=="__main__":
    app.run(main)