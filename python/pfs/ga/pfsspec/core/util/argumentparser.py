import os
import sys
import argparse
from collections.abc import Iterable
from gettext import gettext as _

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, load_config_func=None, **kwargs):
        super().__init__(**kwargs)

        self._load_config_func = load_config_func
        self._exit_on_error = True

    def add_subparser(self):
        pass

    def parse_args(self, args=None, namespace=None):
        if args is None:
            command_args = sys.argv[1:]
        else:
            command_args = args

        # - 1. Try parse command-line to see if the subparser selector positional
        #      arguments are defined. If not, we look for --config arguments, load
        #      the config files and compose the subparser selector arguments manually
        #      based on what's found in the configs.
        try:
            self.set_exit_on_error(enabled=False)
            super().parse_args(command_args, namespace).__dict__
        except argparse.ArgumentError as ex:
            # Retry with reading the subparser choice values from the config file
            self.set_exit_on_error(enabled=True)
            config_paths = self._get_config_paths(command_args)
            paths, configs = self._get_configs(os.getcwd(), config_paths)
            config_args = {}
            for path, config in zip(paths, configs):
                self._merge_args(config_args, os.path.dirname(path), config, override=True, recursive=True)
            subparser_args = self._get_subparser_choices(config_args)
            subparser_args.extend(command_args)
            command_args = subparser_args

        # - 2. Parse command-line args with defaults enabled
        self.set_exit_on_error(enabled=True)
        parsed_args = super().parse_args(args=command_args).__dict__
        paths, configs = self._get_configs(os.getcwd(), parsed_args)
        if len(configs) > 0:
            # If a config file is used:
            # - 3. Load config file, override all specified arguments
            for path, config in zip(paths, configs):
                self._merge_args(parsed_args, os.path.dirname(path), config, override=True, recursive=True)

            # - 4. Reparse command-line with defaults suppressed, apply overrides
            self.disable_defaults()
            override_args = super().parse_args(args=command_args).__dict__
            self._merge_args(parsed_args, os.getcwd(), override_args, override=True, recursive=False)

        return parsed_args

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
            
        return { 'config': paths }

    def _merge_args(self, args, path, other_args, override=True, recursive=False):
        if 'config' in other_args and recursive:
            # This is a config within a config file, load configs recursively, if requested
            paths, configs = self._get_configs(path, other_args)
            for path, config in zip(paths, configs):
                self._merge_args(args, os.path.dirname(path), config, override=override, recursive=True)

        for k in other_args:
            if other_args[k] is not None and (k not in args or args[k] is None or override):
                args[k] = other_args[k]

    def _get_subparser_choices(self, args):
        subparser_args = []
        for a in self._actions:
            if isinstance(a, argparse._SubParsersAction):
                if a.dest in args:
                    subparser_args.append(args[a.dest])
                    # Call resursively for subparsers
                    try:
                        addtl_args = a.choices[args[a.dest]]._get_subparser_choices(args)
                        subparser_args.extend(addtl_args)
                    except KeyError:
                        args = {'parser_name': a.dest,
                                'choices': ', '.join(a.choices)}
                        msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
                        try:
                            raise argparse.ArgumentError(a, msg)
                        except argparse.ArgumentError:
                            err = sys.exc_info()[1]
                            a.error(str(err))

        return subparser_args


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

    def set_exit_on_error(self, enabled=True):
        """
        Disable or enable exit on error
        """
        self._exit_on_error = enabled

        # Call recursively for subparsers
        for a in self._actions:
            if isinstance(a, argparse._SubParsersAction):
                for k in a.choices:
                    if isinstance(a.choices[k], argparse.ArgumentParser):
                        a.choices[k].set_exit_on_error(enabled=enabled)

    def error(self, message):
        if self._exit_on_error:
            super().error(message)
        else:
            raise sys.exc_info()[1]