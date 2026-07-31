
import logging
import os
from pathlib import Path 

def get_logger(log_path=None):
    """
    Get a logger object

    Args:
        log_path (str or Path, optional): The path to the log file. If None, a default path will be used.
    Returns:
        logging.Logger: A logger object configured to write to the specified log file and the console.
    """
    log_level = logging.DEBUG
    if log_path is None:
        log_path = Path(__file__).parent.parent.parent.joinpath("data","tmp","testing_rfi_large_model_training.log")
    if not log_path.parent.exists():
        os.makedirs(log_path.parent)
    logger = logging.getLogger("tests")
    logger.handlers.clear()
    logger.propagate = False
    
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(log_level)
    return logger 
