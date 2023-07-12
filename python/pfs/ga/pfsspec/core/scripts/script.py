import os
import sys
import psutil
import pkgutil, importlib
import json
import yaml
import logging
import numpy as np
import multiprocessing
from multiprocessing import set_start_method
import socket

import pfs.ga.pfsspec.core.util.logging
import pfs.ga.pfsspec   # NOTE: required by module discovery
# TODO: python 3.8 has this built-in, replace import
from ..util import ArgumentParser
import pfs.ga.pfsspec.core.util.shutil as shutil
import pfs.ga.pfsspec.core.util as util
from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.util.notebookrunner import NotebookRunner

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
        self.trace = False
        self.trace_plot_level = Trace.PLOT_LEVEL_NONE
        self.trace_log_level = Trace.LOG_LEVEL_NONE
        self.log_level = None
        self.log_dir = None
        self.log_copy = False
        self.logging_enabled = logging_enabled
        self.logging_console_handler = None
        self.logging_file_handler = None
        self.dump_config = True
        self.dir_history = []
        self.outdir = None
        self.skip_notebooks = False
        self.random_seed = None
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

        self.logger.debug('Looking for script configurations in all modules under the namespace `pfs.ga.pfsspec`.')

        merged_config = {}

        for m in pkgutil.iter_modules(pfs.ga.pfsspec.__path__):
            try:
                modulename = f'pfs.ga.pfsspec.{m.name}.configurations'
                module = importlib.import_module(modulename)
            except ModuleNotFoundError as ex:
                # In case something else is missing we cannot continue
                if ex.msg == f"No module named '{modulename}'":
                    self.logger.debug(f'Module `{m.name}` has no namespace for script configurations.')
                else:
                    raise
                module = None
            if module is not None and hasattr(module, self.CONFIG_NAME):
                config = getattr(module, self.CONFIG_NAME)
                merged_config.update(config)

                self.logger.debug(f'Found script configuration in module `{m.name}` with name {self.CONFIG_NAME}.')
                
        self.parser_configurations = merged_config

        self.logger.debug(f'Final script configuration contains {len(merged_config)} top-level entries: {" ".join(merged_config.keys())}')

    def create_parser(self):
        self.parser = ArgumentParser(load_config_func=Script.load_args_json)
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        # Register two positional variables that determine the plugin class
        # and subclass
        cps = parser.add_subparsers(dest=self.CONFIG_CLASS, required=True)
        for c in self.parser_configurations:
            self.logger.debug(f'Registering sub-parser for class `{self.CONFIG_CLASS}` with value `{c}`')
            cp = cps.add_parser(c)
            sps = cp.add_subparsers(dest=self.CONFIG_SUBCLASS, required=True)
            for s in self.parser_configurations[c]:
                self.logger.debug(f'Registering sub-parser for sub-class `{self.CONFIG_SUBCLASS}` with value `{s}`')
                sp = sps.add_parser(s)

                # Instantiate plugin and register further subparsers
                config = self.parser_configurations[c][s]
                plugin = self.create_plugin(config)
                if plugin is not None:
                    self.logger.debug(f'Instantiated script plugin of type `{plugin.__class__.__name__}`.')
                    subparsers = plugin.add_subparsers(config, sp)
                    if subparsers is not None:
                        self.logger.debug(f'Registering sub-parser for script plugin `{plugin.__class__.__name__}`.')
                        for ss in subparsers:
                            self.add_args(ss, config)
                            plugin.add_args(ss, config)
                    else:
                        self.logger.debug(f'No sub-parser found for script plugin `{plugin.__class__.__name__}`.')
                        self.add_args(sp, config)
                        plugin.add_args(sp, config)
                else:
                    self.logger.debug(f'No script plugin could be created for class `{c}` sub-class `{s}`.')

    def create_plugin(self, config):
        t = config[self.CONFIG_TYPE]
        if t is not None:
            plugin = t()
            return plugin
        else:
            return None

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)

    def add_args(self, parser, config):
        parser.add_argument('--config', type=str, nargs='+', help='Load config from json file.')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode.\n')
        parser.add_argument('--trace', action='store_true', help='Run in trace mode..\n')
        parser.add_argument('--trace-plot-level', type=int, help='Trace plot level.\n')
        parser.add_argument('--trace-log-level', type=int, help='Trace log level.\n')
        parser.add_argument('--profile', action='store_true', help='Run in profiler mode..\n')
        parser.add_argument('--threads', type=int, help='Number of processing threads.\n')
        parser.add_argument('--log-level', type=str, default=None, help='Logging level\n')
        parser.add_argument('--log-dir', type=str, default=None, help='Log directory\n')
        parser.add_argument('--log-copy', action='store_true', help='Copy logfiles to output directory.\n')
        parser.add_argument('--skip-notebooks', action='store_true', help='Skip notebook step.\n')
        parser.add_argument('--random-seed', type=int, default=None, help='Set random seed\n')

    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args()
            
            # Parse some special but generic arguments
            self.debug = self.get_arg('debug', self.debug)
            self.trace = self.get_arg('trace', self.trace)
            self.trace_plot_level = self.get_arg('trace_plot_level', self.trace_plot_level)
            self.trace_log_level = self.get_arg('trace_log_level', self.trace_log_level)
            self.threads = self.get_arg('threads', self.threads)
            self.log_level = self.get_arg('log_level', self.log_level)
            self.log_dir = self.get_arg('log_dir', self.log_dir)
            self.log_copy = self.get_arg('log_copy', self.log_copy)
            self.skip_notebooks = self.get_arg('skip_notebooks', self.skip_notebooks)
            self.random_seed = self.get_arg('random_seed', self.random_seed)

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

    @staticmethod
    def load_args_json(filename):
        with open(filename, 'r') as f:
            args = json.load(f)
        args = Script.resolve_env_vars(args)
        return args

    def dump_env(self, filename):
        with open(filename, 'w') as f:
            for k in os.environ:
                f.write('{}="{}"\n'.format(k, os.environ[k]))

    def dump_command_line(self, filename):
        # Environmental variables, these will be substituted into the arguments
        # to make them shorter, if possible
        vars = Script.get_env_vars()
        
        # See if python script is called from a bash script and retrieve original
        # arguments. This should work for all python entry points called via
        # the `./bin/run` script.
        # TODO: Verify when lib is installed as a package
        # TODO: Make this more generic, /bin/bash is os specific
        parent = psutil.Process().parent()
        if parent is not None and parent.cmdline()[0] == '/bin/bash':
            args = parent.cmdline()[1:]
        else:
            args = sys.argv

        mode = 'a' if os.path.isfile(filename) else 'w'
        with open(filename, mode) as f:
            if mode == 'a':
                f.write('\n')
                f.write('\n')
            for i, arg in enumerate(args):
                arg = self.substitute_env_vars(arg, vars)
                arg = arg.replace('"', '\\"')
                for pattern in [' ']:
                    if pattern in arg:
                        arg = f'\"{arg}\"'
                        break

                if i > 0:
                    f.write(' ')
                f.write(arg)
            f.write('\n')

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

    def get_logging_level(self, pre_init=False):
        if pre_init:
            # If in pre-init stage, the command-line arguments haven't been processed yet,
            # do a coarse processing just to configure logging

            ix = sys.argv.index('--debug') if '--debug' in sys.argv else None
            if ix is not None:
                return logging.DEBUG
            
            ix = sys.argv.index('--log-level') if '--log_level' in sys.argv else None
            if ix is not None:
                return getattr(logging, sys.argv[ix + 1])
        
            return logging.INFO
        else:
            if not self.logging_enabled:
                return logging.FATAL
            elif self.debug:
                return logging.DEBUG
            elif self.log_level is not None:
                return getattr(logging, self.log_level)
            else:
                return logging.INFO

    def setup_logging(self, pre_init=False, logfile=None):
        log_level = self.get_logging_level(pre_init=pre_init)

        # TODO: is this where double logging of multiprocessing comes from?
        # multiprocessing.log_to_stderr(log_level)

        self.logger = logging.getLogger('root')
        self.logger.setLevel(log_level)

        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(asctime)s:%(message)s')

        if self.logging_console_handler is None:
            self.logging_console_handler = logging.StreamHandler(sys.stdout)
            self.logging_console_handler.setLevel(log_level)
            self.logging_console_handler.setFormatter(formatter)
            self.logger.addHandler(self.logging_console_handler)

        if not pre_init:
            if logfile is not None and self.logging_file_handler is None:
                self.logging_file_handler = logging.FileHandler(logfile)
                self.logging_file_handler.setLevel(log_level)
                self.logging_file_handler.setFormatter(formatter)
                self.logger.addHandler(self.logging_file_handler)

        if pre_init:
            self.logger.info('Running script on {}'.format(socket.gethostname()))

    def suspend_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(logging.ERROR)

    def resume_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(self.get_logging_level())
    
    def init_logging(self, outdir):
        logdir = self.log_dir or os.path.join(outdir, 'logs')
        os.makedirs(logdir, exist_ok=True)

        logfile = type(self).__name__.lower() + '.log'
        logfile = os.path.join(logdir, logfile)
        self.setup_logging(logfile=logfile)

        if self.dump_config:
            self.dump_command_line(os.path.join(outdir, 'command.sh'))
            self.dump_env(os.path.join(outdir, 'env.sh'))
            self.dump_args_json(os.path.join(outdir, 'args.json'))

    def execute(self):
        self.prepare()
        self.run()
        self.finish()

    def prepare(self):
        if self.logging_enabled:
            self.setup_logging(pre_init=True)

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

    def run(self):
        raise NotImplementedError()

    def finish(self):
        # Copy logfiles (including tensorboard events)
        if self.log_copy and \
            os.path.abspath(self.outdir) != os.path.abspath(self.log_dir) and \
            os.path.isdir(self.log_dir):

            # TODO: python 3.8 has shutil function for this
            outlogdir = os.path.join(self.outdir, 'logs')
            logging.info('Copying log files to `{}`'.format(outlogdir))
            ignore = None
            shutil.copytree(self.log_dir, outlogdir, ignore=ignore, dirs_exist_ok=True)
        else:
            logging.info('Skipped copying log files to output directory.')

    def execute_notebook(self, notebook_path, output_notebook_path=None, output_html=True, parameters={}, kernel='python3', outdir=None):
        # Note that jupyter kernels in the current env might be different from the ones
        # in the jupyterhub environment

        self.logger.info('Executing notebook {}'.format(notebook_path))

        if outdir is None:
            outdir = self.outdir

        # Project path is added so that the pfsspec lib can be called without
        # installing it
        if 'PROJECT_PATH' not in parameters:
            parameters['PROJECT_PATH'] = os.getcwd()

        if output_notebook_path is None:
            output_notebook_path = os.path.basename(notebook_path)

        nr = NotebookRunner()
        nr.input_notebook = os.path.join('nb', notebook_path + '.ipynb')
        nr.output_notebook = os.path.join(outdir, output_notebook_path + '.ipynb')
        if output_html:
            nr.output_html = os.path.join(outdir, output_notebook_path + '.html')
        nr.parameters = parameters
        nr.kernel = kernel
        nr.run()