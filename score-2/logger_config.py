"""
Author: Hanlong Chen
Version: 1.0
Contact: hanlong@ucla.edu
"""

import logging
import os
import sys
import colorlog

default_level = logging.WARNING

def is_INFO_mode():
    return sys.gettrace() is not None

log_level_env = os.environ.get('LOGLEVEL', '').lower()

if is_INFO_mode():
    IS_DEBUG = True
    IS_INFO = True
else:
    IS_DEBUG = False
    IS_INFO = False

if log_level_env == 'debug':
    IS_DEBUG = True
    IS_INFO = True
elif log_level_env == 'info':
    IS_DEBUG = False
    IS_INFO = True

log_format = '%(log_color)s%(asctime)s - %(module)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
color_formatter = colorlog.ColoredFormatter(
    log_format,
    datefmt=date_format,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': '',
        'WARNING': 'yellow',
        'ERROR': 'bold_red',
        'CRITICAL': 'bg_bold_red',
    }
)

handler = logging.StreamHandler()
handler.setFormatter(color_formatter)
logging.basicConfig(level=default_level, handlers=[handler])

def get_logger(name):
    logger = logging.getLogger(name)
    if IS_DEBUG:
        logger.setLevel(logging.DEBUG)
    elif IS_INFO:
        logger.setLevel(logging.INFO)

    return logger

if __name__ == "__main__":
    logger = get_logger(__name__)
    # logger.trace("This is a TRACE message")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")