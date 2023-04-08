import os

from registry import Registry
from utils.di_module_utils import create_prefix_module_pattern, load_config_modules

_METHODS_PATTERN = create_prefix_module_pattern('m')

load_config_modules(os.path.dirname(__file__), _METHODS_PATTERN)

def get_method(name: str):
    registry = Registry()
    return registry.get_from_registry(f'methods/{name}')
