# src/utils.py

import os
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_level="INFO"):
    """
    Configure a simple logging format to stdout.
    """
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger()

def ensure_directories_exist(*dirs):
    """
    Given a list of directory paths, create each if it does not exist.
    """
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
                logger.debug(f"Created directory: {d}")
            except Exception as e:
                logger.error(f"Could not create directory {d}: {e}")
                raise