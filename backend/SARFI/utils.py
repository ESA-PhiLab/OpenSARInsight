import logging
import logging.handlers
import os
from pathlib import Path

class SARFILogger:
    def __init__(self, log_out_path=None):
        self.log_out_path = log_out_path
        self.logger:logging.Logger = self._setup_logger()

    def update_log_level(self, level=logging.DEBUG):
        valid_levels = [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR]
        if  level not in valid_levels:
            raise ValueError(f"level invalid, must be one of {valid_levels}")
        self.logger.setLevel(level)

    def _setup_logger(self):
        """ Configure rotating file handler """
        logger = logging.getLogger("sarfi_logger")

        if self.log_out_path is None:
            self.log_out_path = Path(__file__).parent.joinpath("tmp","logs","sarfi_3.log")
            if not self.log_out_path.parent.exists():
                os.makedirs(self.log_out_path.parent, exist_ok=False)
        rotating_handler = logging.handlers.RotatingFileHandler(
            self.log_out_path, maxBytes=1_000_000, backupCount=5
        )

        # Configure basic logging to console and rotating file
        logger.level = logging.INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        rotating_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(rotating_handler)
        logger.addHandler(console_handler)

        return logger
