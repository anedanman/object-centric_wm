from abc import ABC, abstractmethod
from typing import Optional

import gym

from data_collection.configs.collect_configs import BaseCollectConfig
from data_collection.constants import Environments
from data_collection.utils import get_torch_device


class BaseAgent(ABC):

    def __init__(self,
                 env: gym.Env,
                 env_name: Environments,
                 collect_config: BaseCollectConfig,
                 device: Optional[str] = None):
        self.env = env
        self.action_space = env.action_space
        self.torch_device = get_torch_device(device)
        self.collect_config = collect_config
        self.env_name = env_name
    @abstractmethod
    def get_action(self, obs, is_burnin_phase=False):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()