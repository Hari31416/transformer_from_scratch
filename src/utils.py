import json

import logging
from typing import List, Optional, Union, Dict
from inspect import signature


def set_logger_level_to_all_local(level: int) -> None:
    """Sets the level of all local loggers to the given level.

    Parameters
    ----------
    level : int, optional
        The level to set the loggers to, by default logging.DEBUG.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]

    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if hasattr(logger, "local"):
                logger.setLevel(level)


def create_simple_logger(
    logger_name: str, level: str = "info", set_level_to_all_loggers: bool = False
) -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical".

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.local = True
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if set_level_to_all_loggers:
        set_logger_level_to_all_local(level)
    return logger


logger = create_simple_logger(__name__)


def get_parameters_list(func):
    optional_params = [
        p.name for p in signature(func).parameters.values() if p.default != p.empty
    ]
    non_optional_params = [
        p.name for p in signature(func).parameters.values() if p.default == p.empty
    ]
    # remove special parameters like self
    if "self" in non_optional_params:
        non_optional_params.remove("self")
    return non_optional_params, optional_params


class Config:
    """An abstract class for configuration classes. A configuration class has two main attributes:

    - ALLOWED_KEYS: a list of allowed keys
    - ConfigFor: the class that this config is for

    Using this class, we can create a configuration class for any class by inheriting from it and setting the ALLOWED_KEYS and ConfigFor attributes.
    """

    ALLOWED_KEYS: List[str] = None  # a list of allowed keys
    ConfigFor: Union[object, str, None] = None  # the class that this config is for

    def __init__(self, **kwargs):

        # check if all the keys are allowed
        keys_provided = list(kwargs.keys())
        if self.ALLOWED_KEYS is None:
            delta = []
        else:
            delta = list(set(keys_provided) - set(self.ALLOWED_KEYS))
        if len(delta) > 0:
            msg = f"Provided keys not allowed: {delta}"
            logger.error(msg)
            raise ValueError(msg)

        self.__dict__.update(kwargs)

    def to_dict(self):
        if self.ALLOWED_KEYS is None:
            return self.__dict__

        all_variables = list(self.__dict__.keys())
        available_keys = self.ALLOWED_KEYS
        intersection = list(set(all_variables) & set(available_keys))
        return {k: self.__dict__[k] for k in intersection}

    @staticmethod
    def load_from_path(path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return Config(**d)

    def load_object(self, obj: Optional[object] = None):
        # if ConfigFor is not set, and obj is not provided, we can't create the object
        if self.ConfigFor is None and obj is None:
            msg = "ConfigFor is not set, and obj is not provided"
            logger.error(msg)
            raise ValueError(msg)

        # if no object is provided, we will use ConfigFor
        if obj is None:
            # if ConfigFor is a string, we will use the global object with that name
            if isinstance(self.ConfigFor, str):
                logger.info("No object provided, Getting object from globals")
                obj = globals()[self.ConfigFor]
            else:
                logger.info("No object provided, using ConfigFor")
                obj = self.ConfigFor
            logger.debug(f"Found object: {obj.__class__.__name__}")

        # check if all non-optional parameters are available before creating the object
        non_optional_params, optional_params = get_parameters_list(obj.__init__)
        params_dict = self.to_dict()
        parameters_available = list(params_dict.keys())

        delta_non_optional = list(set(non_optional_params) - set(parameters_available))
        if len(delta_non_optional) > 0:
            msg = f"Missing non-optional parameters: {delta_non_optional}"
            logger.error(msg)
            raise ValueError(msg)

        # for optional parameters, we can ignore them if they are not available
        delta_optional = list(set(optional_params) - set(parameters_available))
        if len(delta_optional) > 0:
            msg = f"Missing optional parameters: {delta_optional}"
            logger.warning(msg)

        return obj(**self.to_dict())

    def save(self, path: str):
        # not that this may always work for some classes
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def to_dict_serializable(self):
        """Converts the configuration to a dictionary that can be serialized to JSON."""
        params = self.to_dict().copy()
        for k, v in params.items():
            if isinstance(v, Config):
                params[k] = v.to_dict_serializable()
            if not isinstance(v, (int, float, str, list, dict)):
                try:
                    logger.warning(
                        f"Non-serializable value found for {k}. Only class name will be saved."
                    )
                    params[k] = v.__class__.__name__
                except Exception as e:
                    logger.warning(
                        f"Error getting class name for {k}. It will be kept empty."
                    )
                    params[k] = None
            else:
                params[k] = v

        return params
