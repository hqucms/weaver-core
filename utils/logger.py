import logging
import sys
import os
from functools import lru_cache


def _configLogger(name, filename=None, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stdout
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)
    if filename:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)


_logger = logging.getLogger('weaver')


@lru_cache(10)
def warn_once(msg, logger=_logger):
    # Keep track of 10 different messages and then warn again
    logger.warning(msg)
