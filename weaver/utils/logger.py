import logging
import sys
import os


def _configLogger(name, stdout=sys.stdout, filename=None, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stdout
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    if stdout:
        console = logging.StreamHandler(stdout)
        console.setLevel(loglevel)
        console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(console)
    if filename:
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(os.path.dirname(filename))
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)


class ColoredLogger():
    color_dict = {
        'black': '\033[0;30m',
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'orange': '\033[0;33m',
        'blue': '\033[0;34m',
        'purple': '\033[0;35m',
        'cyan': '\033[0;36m',
        'lightgray': '\033[0;37m',

        'darkgray': '\033[1;30m',
        'lightred': '\033[1;31m',
        'lightgreen': '\033[1;32m',
        'yellow': '\033[1;33m',
        'lightblue': '\033[1;34m',
        'lightpurple': '\033[1;35m',
        'lightcyan': '\033[1;36m',
        'white': '\033[1;37m',

        'bold': '\033[1m',
        'endcolor': '\033[0m',
    }

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def colorize(self, msg, color):
        return self.color_dict[color] + msg + self.color_dict['endcolor']

    def debug(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.error(msg, *args, **kwargs)


_logger = ColoredLogger('weaver')

_warning_counter = {}


def warn_n_times(msg, n=10, logger=_logger):
    if msg not in _warning_counter:
        _warning_counter[msg] = 0
    if _warning_counter[msg] < n:
        logger.warning(msg)
    _warning_counter[msg] += 1
