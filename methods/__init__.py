from registry import Registry


def register_method(name: 'str'):
    registry = Registry()

    def inner(target):
        registry.add_to_registry(f'methods/{name}', target)
        return target

    return inner


def get_method(name: 'str'):
    registry = Registry()
    return registry.get_from_registry(f'methods/{name}')