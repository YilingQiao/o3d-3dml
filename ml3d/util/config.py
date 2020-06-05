import os.path
import shutil
import sys
import tempfile
from importlib import import_module


class Config(object):
    """docstring for Config"""
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict shoud be a dict, but'
                            f'got {type(cfg_dict)}')

        self.cfg_dict = cfg_dict

    @staticmethod
    def load_from_file(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f'File {filename} not found')

        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
                temp_config_name = os.path.basename(temp_config_file.name)
                shutil.copyfile(filename,
                                os.path.join(temp_config_dir, temp_config_name))
                temp_module_name = os.path.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()

        return Config(cfg_dict)

    

        