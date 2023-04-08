import os

from utils.di_module_utils import create_prefix_module_pattern, load_config_modules
from registry.registry import Registry

_DATASET_PATTERN = create_prefix_module_pattern('ds')

load_config_modules(os.path.dirname(__file__), _DATASET_PATTERN)

def get_dataset(name: str):
    registry = Registry()
    return registry.get_from_registry(f'datasets/{name}')
