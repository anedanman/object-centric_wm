from os.path import dirname, basename, isfile, join
import glob


def get_all_modules(file):
    modules = glob.glob(join(dirname(file), "*.py"))
    return [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
