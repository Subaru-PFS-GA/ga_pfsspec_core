import os
import sys
import pkgutil, importlib
import json
import yaml
import logging
import argparse
import numpy as np
import multiprocessing
from multiprocessing import set_start_method
import socket
from collections.abc import Iterable

import pfsspec
import pfsspec.core.util as util
from pfsspec.core.util.notebookrunner import NotebookRunner

class Script():

    CONFIG_NAME = None
    CONFIG_CLASS = 'class'
    CONFIG_SUBCLASS = 'subclass'
    CONFIG_TYPE = 'type'

    def __init__(self, logging_enabled=True):

        # Spawning worker processes is slower but might help with deadlocks
        # multiprocessing.set_start_method("fork")

        self.parser = None
        self.args = None
        self.debug = False
        self.random_seed = None
        self.log_level = None
        self.log_dir = None
        self.logging_enabled = logging_enabled
        self.logging_console_handler = None
        self.logging_file_handler = None
        self.dir_history = []
        self.outdir = None
        self.is_batch = 'SLURM_JOBID' in os.environ
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            self.threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            self.threads = multiprocessing.cpu_count() // 2
        
    def find_configurations(self):
        """
        Returns a list of configuration dictionaries found in a list of
        modules. The list is used to create a combined configuration to
        initialize the parser modules.
        """

        merged_config = {}

        for m in pkgutil.iter_modules(pfsspec.__path__):
            module = importlib.import_module('pfsspec.{}.configurations'.format(m.name))
            if hasattr(module, self.CONFIG_NAME):
                config = getattr(module, self.CONFIG_NAME)
                merged_config.update(config)

        self.parser_configurations = merged_config

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        # Register two positional variables that determine the plugin class
        # and subclass
        cps = parser.add_subparsers(dest=self.CONFIG_CLASS)
        for c in self.parser_configurations:
            cp = cps.add_parser(c)
            sps = cp.add_subparsers(dest=self.CONFIG_SUBCLASS)
            for s in self.parser_configurations[c]:
                sp = sps.add_parser(s)

                # Instantiate plugin and register further subparsers
                plugin = self.create_plugin(self.parser_configurations[c][s])
                subparsers = plugin.add_subparsers(self.parser_configurations[c][s], sp)
                if subparsers is not None:
                    for ss in subparsers:
                        self.add_args(ss)
                        plugin.add_args(ss)
                else:
                    self.add_args(sp)
                    plugin.add_args(sp)

    def create_plugin(self, config):
        return config[self.CONFIG_TYPE]()

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--config', type=str, nargs='+', help='Load config from json file.')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode\n')
        parser.add_argument('--threads', type=int, help='Number of processing threads.\n')
        parser.add_argument('--log-level', type=str, default=None, help='Logging level\n')
        parser.add_argument('--log-dir', type=str, default=None, help='Log directory\n')
        parser.add_argument('--random-seed', type=int, default=None, help='Set random seed\n')

    def get_configs(self, args):
        configs = []
        if 'config' in args and args['config'] is not None:
            if isinstance(args['config'], Iterable):
                filenames = list(args['config'])
            else:
                filenames = [args['config']]

            for filename in filenames:
                config = self.load_args_json(filename)
                configs.append(config)

        return configs

    def parse_args(self):
        if self.args is None:
            # - 1. parse command-line args with defaults enabled (already done above)
            self.args = self.parser.parse_args().__dict__
            configs = self.get_configs(self.args)
            if len(configs) > 0:
                # If a config file is used:
                # - 2. load config file, override all specified arguments
                for config in configs:
                    self.merge_args(config, override=True, recursive=True)

                # - 3. reparse command-line with defaults suppressed, apply overrides
                self.disable_parser_defaults(self.parser)
                command_args = self.parser.parse_args().__dict__
                self.merge_args(command_args, override=True, recursive=False)

            # Parse some special but generic arguments
            self.debug = self.get_arg('debug', self.debug)
            self.threads = self.get_arg('threads', self.threads)
            self.log_level = self.get_arg('log_level', self.log_level)
            self.log_dir = self.get_arg('log_dir', self.log_dir)
            self.random_seed = self.get_arg('random_seed', self.random_seed)

    def merge_args(self, other_args, override=True, recursive=False):
        if 'config' in other_args and recursive:
            # This is a config within a config file, load configs recursively, if requested
            configs = self.get_configs(other_args)
            for config in configs:
                self.merge_args(config, override=override, recursive=True)

        for k in other_args:
            if other_args[k] is not None and (k not in self.args or self.args[k] is None or override):
                self.args[k] = other_args[k]

    def disable_parser_defaults(self, parser):
        # Call recursively for subparsers
        for a in parser._actions:
            if isinstance(a, (argparse._StoreAction, argparse._StoreConstAction,
                              argparse._StoreTrueAction, argparse._StoreFalseAction)):
                a.default = None
            elif isinstance(a, argparse._SubParsersAction):
                for k in a.choices:
                    if isinstance(a.choices[k], argparse.ArgumentParser):
                        self.disable_parser_defaults(a.choices[k])

    @staticmethod
    def get_env_vars(prefix='PFSSPEC'):
        vars = {}
        for k in os.environ:
            if k.startswith(prefix):
                vars[k] = os.environ[k]
        return vars

    @staticmethod
    def substitute_env_vars(data, vars=None):
        vars = vars or Script.get_env_vars()

        if isinstance(data, dict):
            return {k: Script.substitute_env_vars(data[k], vars) for k in data}
        elif isinstance(data, list):
            return [Script.substitute_env_vars(d, vars) for d in data]
        elif isinstance(data, tuple):
            return tuple([Script.substitute_env_vars(d, vars) for d in data])
        elif isinstance(data, str):
            for k in vars:
                data = data.replace(vars[k], '${' + k + '}')
            return data
        else:
            return data

    @staticmethod
    def resolve_env_vars(data, vars=None):
        vars = vars or Script.get_env_vars()

        if isinstance(data, dict):
            return {k: Script.resolve_env_vars(data[k], vars) for k in data}
        elif isinstance(data, list):
            return [Script.resolve_env_vars(d, vars) for d in data]
        elif isinstance(data, tuple):
            return tuple([Script.resolve_env_vars(d, vars) for d in data])
        elif isinstance(data, str):
            for k in vars:
                data = data.replace('${' + k + '}', vars[k])
            return data
        else:
            return data

    @staticmethod
    def dump_json_default(obj):
        if isinstance(obj, float):
            return "%.5f" % obj
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                if obj.size < 100:
                    return obj.tolist()
                else:
                    return "(not serialized)"
            else:
                return obj.item()
        return "(not serialized)"

    def dump_json(self, obj, filename):
        with open(filename, 'w') as f:
            if type(obj) is dict:
                json.dump(obj, f, default=Script.dump_json_default, indent=4)
            else:
                json.dump(obj.__dict__, f, default=Script.dump_json_default, indent=4)

    def dump_args_json(self, filename):
        args = Script.substitute_env_vars(self.args)
        with open(filename, 'w') as f:
            json.dump(args, f, default=Script.dump_json_default, indent=4)

    def dump_args_yaml(self, filename):
        args = Script.substitute_env_vars(self.args)
        with open(filename, 'w') as f:
            yaml.dump(args, f, indent=4)

    def load_args_json(self, filename):
        with open(filename, 'r') as f:
            args = json.load(f)
        args = Script.resolve_env_vars(args)
        return args

    def dump_env(self, filename):
        with open(filename, 'w') as f:
            for k in os.environ:
                f.write('{}="{}"\n'.format(k, os.environ[k]))

    def load_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def create_output_dir(self, dir, resume=False):
        self.logger.info('Output directory is {}'.format(dir))
        if resume:
            if os.path.exists(dir):
                self.logger.info('Found output directory.')
            else:
                raise Exception("Output directory doesn't exist, can't continue.")
        elif os.path.exists(dir):
            if len(os.listdir(dir)) != 0:
                raise Exception('Output directory is not empty: `{}`'.format(dir))
        else:
            self.logger.info('Creating output directory {}'.format(dir))
            os.makedirs(dir)

    def pushd(self, dir):
        self.dir_history.append(os.getcwd())
        os.chdir(dir)

    def popd(self):
        os.chdir(self.dir_history[-1])
        del self.dir_history[-1]

    def get_logging_level(self):
        if not self.logging_enabled:
            return logging.FATAL
        elif self.debug:
            return logging.DEBUG
        elif self.log_level is not None:
            return getattr(logging, self.log_level)
        else:
            return logging.INFO

    def setup_logging(self, logfile=None):
        # TODO: is this where double logging of multiprocessing comes from?
        if self.logging_enabled:
            multiprocessing.log_to_stderr(self.get_logging_level())

        self.logger = logging.getLogger()
        self.logger.setLevel(self.get_logging_level())

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if logfile is not None and self.logging_file_handler is None:
            self.logging_file_handler = logging.FileHandler(logfile)
            self.logging_file_handler.setLevel(self.get_logging_level())
            self.logging_file_handler.setFormatter(formatter)
            self.logger.addHandler(self.logging_file_handler)

        if self.logging_console_handler is None:
            self.logging_console_handler = logging.StreamHandler(sys.stdout)
            self.logging_console_handler.setLevel(self.get_logging_level())
            self.logging_console_handler.setFormatter(formatter)
            self.logger.addHandler(self.logging_console_handler)

        self.logger.info('Running script on {}'.format(socket.gethostname()))

    def suspend_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(logging.ERROR)

    def resume_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(self.get_logging_level())

    def save_command_line(self, filename):
        mode = 'a' if os.path.isfile(filename) else 'w'
        with open(filename, mode) as f:
            if mode == 'a':
                f.write('\n')
                f.write('\n')
            f.write(' '.join(sys.argv))
    
    def init_logging(self, outdir):
        logdir = self.log_dir or os.path.join(outdir, 'logs')
        os.makedirs(logdir, exist_ok=True)

        logfile = type(self).__name__.lower() + '.log'
        logfile = os.path.join(logdir, logfile)
        self.setup_logging(logfile)

        self.save_command_line(os.path.join(outdir, 'command.sh'))
        self.dump_env(os.path.join(outdir, 'env.sh'))
        self.dump_args_json(os.path.join(outdir, 'args.json'))

    def execute(self):
        self.prepare()
        self.run()
        self.finish()

    def prepare(self):
        self.find_configurations()
        self.create_parser()
        self.parse_args()

        # TODO: fix this by moving all initializations from inherited classes
        #       to here. This will require testing all scripts.
        # Turn on logging to the console. This will be re-configured once the
        # output directory is created
        if self.logging_enabled:
            self.setup_logging()

        if self.debug:
            np.seterr(divide='raise', over='raise', invalid='raise')
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def run(self):
        raise NotImplementedError()

    def finish(self):
        pass

    def execute_notebook(self, notebook_name, output_notebook_name=None, output_html=True, parameters={}, kernel='python3', outdir=None):
        # Note that jupyter kernels in the current env might be different from the ones
        # in the jupyterhub environment

        self.logger.info('Executing notebook {}'.format(notebook_name))

        if outdir is None:
            outdir = self.args['out']

        # Project path is added so that the pfsspec lib can be called without
        # installing it
        if 'PROJECT_PATH' not in parameters:
            parameters['PROJECT_PATH'] = os.getcwd()

        if output_notebook_name is None:
            output_notebook_name = notebook_name

        nr = NotebookRunner()
        nr.input_notebook = os.path.join('nb', notebook_name + '.ipynb')
        nr.output_notebook = os.path.join(outdir, output_notebook_name + '.ipynb')
        if output_html:
            nr.output_html = os.path.join(outdir, output_notebook_name + '.html')
        nr.parameters = parameters
        nr.kernel = kernel
        nr.run()