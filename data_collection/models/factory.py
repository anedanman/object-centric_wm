from pathlib import Path
from typing import Optional

import torch

from data_collection.configs.a3c_configs import get_a3c_config
from data_collection.constants import Environments
from data_collection.models.a3c_baby import NNPolicy
from data_collection.utils import get_torch_device


def get_model(env: Environments,
              weights_path: Optional[str] = None,
              device_option: Optional[str] = None) -> NNPolicy:

    config = get_a3c_config(env)
    model = NNPolicy(**config.__dict__)
    if weights_path:
        model.try_load(weights_path)
    device = get_torch_device(device_option)
    model.to(device)
    return model
