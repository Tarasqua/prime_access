"""
Load entrance detection config data
"""

import os
import yaml
from functools import reduce
from pathlib import Path


class EntranceConfig:
    """Config data loader"""

    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yml')
        assert os.path.exists(config_path), "Config not found"
        with open(config_path, 'r') as f:
            self.config_loader: dict = yaml.load(f, Loader=yaml.FullLoader)

    def get(self, *setting_name):
        """Config getter"""
        return reduce(dict.get, setting_name, self.config_loader)
