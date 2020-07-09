import logging
from logging.handlers import RotatingFileHandler

LOGFILE = "logger.out"


def get_configured_logger(name: str) -> logging.Logger:
    """
    Returns a logger that prints both to console and a file specified within
    config
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # create console and file handler and set level to debug
    ch = logging.StreamHandler()

    fh = RotatingFileHandler(LOGFILE, maxBytes=2000)

    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name) - %(levelname)s - %(message)s")

    # add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger