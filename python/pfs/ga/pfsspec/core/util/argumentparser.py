import os
import sys
import argparse
from collections.abc import Iterable

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, load_config_func=None, **kwargs):
        super().__init__(**kwargs)

        self._load_config_func = load_config_func

    def parse_args(self, args=None, namespace=None):
        # - 1. parse command-line args with defaults enabled
        args = super().parse_args().__dict__
        paths, configs = self._get_configs(os.getcwd(), args)
        if len(configs) > 0:
            # If a config file is used:
            # - 2. load config file, override all specified arguments
            for path, config in zip(paths, configs):
                self._merge_args(args, os.path.dirname(path), config, override=True, recursive=True)

            # - 3. reparse command-line with defaults suppressed, apply overrides
            self.disable_defaults()
            command_args = super().parse_args().__dict__
            self._merge_args(args, os.getcwd(), command_args, override=True, recursive=False)

        return args

    def _get_configs(self, path, args):
        configs = []
        paths = []
        if 'config' in args and args['config'] is not None:
            if isinstance(args['config'], Iterable):
                filenames = list(args['config'])
            else:
                filenames = [args['config']]

            for filename in filenames:
                fn = os.path.join(path, filename)
                config = self._load_config_func(fn)
                paths.append(fn)
                configs.append(config)

        return paths, configs

    def _get_config_paths(self, arg_strings):
        """
        Find --config in arg_strings
        """
        paths = None
        for i, arg in enumerate(arg_strings):
            if paths is not None and arg[0] == '-':
                # Stop if new option after --config is found
                break
            elif arg == '--config':
                paths = []
            elif paths is not None:
                paths.append(arg)
            
        return paths

    def _merge_args(self, args, path, other_args, override=True, recursive=False):
        if 'config' in other_args and recursive:
            # This is a config within a config file, load configs recursively, if requested
            paths, configs = self._get_configs(path, other_args)
            for path, config in zip(paths, configs):
                self._merge_args(args, os.path.dirname(path), config, override=override, recursive=True)

        for k in other_args:
            if other_args[k] is not None and (k not in args or args[k] is None or override):
                args[k] = other_args[k]

    def disable_defaults(self):
        """
        Disable default values for each argument.
        """

        # Call recursively for subparsers
        for a in self._actions:
            if isinstance(a, (argparse._StoreAction, argparse._StoreConstAction,
                              argparse._StoreTrueAction, argparse._StoreFalseAction)):
                a.default = None
            elif isinstance(a, argparse._SubParsersAction):
                for k in a.choices:
                    if isinstance(a.choices[k], argparse.ArgumentParser):
                        a.choices[k].disable_defaults()