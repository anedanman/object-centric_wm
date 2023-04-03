from typing import Any

from .singleton import Singleton


class Registry(Singleton):
    _registry = {}

    def add_to_registry(self, token: str, val: Any):
        if token in self._registry:
            raise ValueError("Token already in registry")
        self._registry[token] = val

    def get_from_registry(self, token: str):
        return self._registry[token]
