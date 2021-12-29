#!/bin/bash

pip install -r requirements.txt
pip install -e ./

export PATH=$PATH:~/.local/bin

# we can execute this on the head node
# wandb login $1 

# sweep agent:
wandb agent $1

# regular training:
# python dlcomp/train.py --config experiments-cfg/autoencoder_3.yaml
