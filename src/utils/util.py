import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch
import pandas as pd

import os
import yaml
from time import gmtime, strftime

import wandb
from pytorch_lightning.loggers import CSVLogger, WandbLogger

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_logger(use_wandb, sweep_mode, output_dir):
    my_loggers = []
    csv_logger = CSVLogger(save_dir=output_dir, name="result")
    my_loggers.append(csv_logger)

    if use_wandb:
        if sweep_mode:
            # When using sweep mode, we don't need to call wandb.init() here.
            wandb_logger = WandbLogger(
                save_dir=output_dir,
                # name=os.path.basename(output_dir),
                # project="sketch classification",
                job_type="sweep"
            )
        else:
            # Regular initialization for non-sweep mode
            wandb.init(
                project="sketch classification",
                entity="nav_sketch",
                name=output_dir.replace("./result/", ""),
            )
            wandb_logger = WandbLogger(
                save_dir=output_dir,
                name=os.path.basename(output_dir),
                project="sketch classification",
            )
        my_loggers.append(wandb_logger)

    return my_loggers