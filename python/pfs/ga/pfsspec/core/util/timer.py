import time
import logging

from ..setup_logger import logger

class Timer:
    def __init__(self, message, logger=logger, level=logging.DEBUG, print_to_console=False, print_to_log=True):
        self.logger = logger
        self.message = message
        self.level = level
        self.print_to_console = print_to_console
        self.print_to_log = print_to_log

        self.start = time.perf_counter()

    def __enter__(self):
        self.banner()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stamp()

    def banner(self):
        if self.print_to_console:
            print(self.message)
        if self.print_to_log:
            self.logger.log(self.level, self.message)

    def stamp(self):
        elapsed = time.perf_counter() - self.start
        msg = '... done in {} sec.'.format(elapsed)
        if self.print_to_console:
            print(msg)
        if self.print_to_log:
            self.logger.log(self.level, msg)
