import logging
import os

def setup_logger(name, log_dir="logs", level=logging.INFO):
    """
    Sets up a logger with the given name and log directory.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler for logging to a file
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Adding handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
