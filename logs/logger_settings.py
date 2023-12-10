"""
Script to configure Python logger for model building
"""

import logging

LOGGING_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOGGING_LEVEL,
    filename="./logs/cnn_logs.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

USE_LOGGING = True
