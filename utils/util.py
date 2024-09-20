import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch
import pandas as pd


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
