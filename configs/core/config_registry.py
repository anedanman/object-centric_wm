from registry import Registry


def register_config(name: str):
    def decorator(config_getter):
        registry = Registry()
        registry.add_to_registry(f'config/{name}', config_getter)
        return config_getter

    return decorator


def get_config(name: str):
    registry = Registry()
    return registry.get_from_registry(f'config/{name}')()


def create_category(category: str):
    def register_in_category(name: str):
        return register_config(f'{category}/{name}')

    def get_from_category(name: str):
        return get_config(f'{category}/{name}')

    return register_in_category, get_from_category
