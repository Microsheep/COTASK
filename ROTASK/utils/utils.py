import os
import sys
import logging
import logging.config

from typing import Optional

# Initiate Logger
logger = logging.getLogger(__name__)


def uncaught_exception_handler(exc_type, exc_value, exc_traceback):
    # https://stackoverflow.com/a/16993115
    if issubclass(exc_type, KeyboardInterrupt):
        # KeyboardInterrupt won't be catched into logger
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught Exception:", exc_info=(exc_type, exc_value, exc_traceback))


def setup_logging(log_path: Optional[str] = None, level: str = "DEBUG"):
    handlers_dict = {
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    }

    if log_path is not None:
        safe_dir(log_path, with_filename=True)
        handlers_dict["file_handler"] = {
            "class": "logging.FileHandler",
            "formatter": "full",
            "level": "DEBUG",
            "filename": log_path,
            "encoding": "utf8"
        }

    # Configure logging
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[ %(asctime)s ] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "full": {
                "format": "[ %(asctime)s ] %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": handlers_dict,
        "loggers": {
            "ROTASK": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "__main__": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
        }
    }

    # Deal with dual log issue
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger().handlers[0].setLevel(logging.WARNING)

    logging.config.dictConfig(config_dict)
    logger.info("Setup Logging!")

    # Catch all Uncaught Exceptions
    # TODO: Seems to not be working correctly due to wandb
    sys.excepthook = uncaught_exception_handler


def safe_dir(path: str, with_filename: bool = False) -> str:
    dir_path = os.path.dirname(path) if with_filename else path
    if not os.path.exists(dir_path):
        logger.info("Dir %s not exist, creating directory!", dir_path)
        os.makedirs(dir_path)
    return os.path.abspath(path)
