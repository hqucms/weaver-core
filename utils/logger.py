import logging
import sys


def _configLogger(name, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stdout
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)


_logger = logging.getLogger('NNTools')
_configLogger('NNTools')
