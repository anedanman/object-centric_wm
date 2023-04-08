import importlib.util
import os
import re
from typing import AnyStr, Pattern


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("config_module", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def create_prefix_module_pattern(prefix: str):
    return re.compile(fr'^{prefix}_.*(?<!_base)\.py$')


def load_config_modules(dirname: str, regex: Pattern[AnyStr]):
    for entry in os.listdir(dirname):
        if regex.match(entry) is not None:
            load_module(os.path.join(dirname, entry))

