from utils.get_all_modules import get_all_modules
from registry.registry import Registry

__all__ = get_all_modules(__file__)


def register_dataset(name: str):
    def decorator(dataset_getter):
        registry = Registry()
        registry.add_to_registry(f'datasets/{name}', dataset_getter)
        return dataset_getter

    return decorator


def get_dataset(name: str):
    registry = Registry()
    return registry.get_from_registry(f'datasets/{name}')
