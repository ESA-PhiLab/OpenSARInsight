"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: logging_setup.py

Description:
    This module provides a logging configuration for the project. It configures the Python root logger to write timestamped log files 
    under the "logs/" directory and also outputs logs to the console.

History:
    - 2025-08-07:
        This is the first version of the logging setup code.
    - 2025-08-18:
        Added the required comments.
      
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-08-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime

__all__ = ["init_logging"]

_LOG_DIR      = "logs"                              
_FORMAT       = "%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
_DATEFMT      = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LVL  = logging.INFO


def _timestamped_logfile(log_dir: Path) -> Path:
    """
        Generate a unique logfile path.

        Args:
            log_dir (Path): Directory in which the log file should be created.

        Returns:
            Path: Full path of the logfile with a timestamped name.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return log_dir / f"logs_{stamp}.log"


def init_logging(
    *,
    log_dir: str | Path = _LOG_DIR,
    level: int = _DEFAULT_LVL,
) -> None:
    """
        Configure the root logger once per process.

        Args:
            log_dir (str | Path): Directory where timestamped log files will be stored.
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

        Returns:
            None
    """
    root = logging.getLogger()
    if root.handlers:              
        return

    logfile = _timestamped_logfile(Path(log_dir))

    logging.basicConfig(
        filename=logfile,
        filemode="w",            
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
    )

    # Optional: also echo to console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)
