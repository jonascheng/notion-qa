import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(name)s %(filename)s %(lineno)s %(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "loggers": {
        "langchain": {
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False
        },
        "langchain_openai": {
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False
        },
        "index": {
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False
        },
        "query": {
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False
        },
    },
    "root": {
        "handlers": ["stdout"],
        "level": "INFO"
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
