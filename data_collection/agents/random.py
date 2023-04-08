from typing import Optional

import gym

from data_collection.agents.base import BaseAgent
from data_collection.configs.collect_configs import BaseCollectConfig
from data_collection.constants import Environments


class RandomAgent(BaseAgent):
    def __init__(self, env: gym.Env, env_name: Environments, collect_config: BaseCollectConfig,
                 device: Optional[str] = None):
        super().__init__(env, env_name, collect_config, device)

    def get_action(self, obs, is_burnin_phase=False):
        return self.action_space.sample()

    def reset(self):
        pass