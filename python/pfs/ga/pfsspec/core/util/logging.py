import logging

# Add trace level and method

def add_level(level_name = 'TRACE', level_num = logging.DEBUG - 5):
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), level_name.lower(), logForLevel)
    setattr(logging, level_name.lower(), logToRoot)

add_level()