#!/usr/bin/env python3
"""
Logger for GridWorld
"""

# ---------- Future Import ----------
from __future__ import annotations

# ---------- Standard Library Imports ----------
import logging
from datetime import datetime
import json

# ---------- Third Party Imports ----------


# ---------- Project Level Imports ----------


# ---------- Begin File ----------

class MyLogger:
    """Logger class for GridWorld"""

    def __init__(self, name: str = "gridworld") -> None:
        self.name = name
        self.logger = logging.getLogger(name)

    def info(self, message: str, extra: dict = None) -> None:
        """Log info message with optional extra fields"""
        if extra is None:
            extra = {}
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "logger": self.name,
            "message": message,
        }
        log_data.update(extra)
        json_str = json.dumps(log_data)
        self.logger.info(json_str)
        print(json_str)

    def error(self, message: str, extra: dict = None, exc_info: bool = False) -> None:
        """Log error message with optional extra fields and exception traceback"""
        if extra is None:
            extra = {}
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "logger": self.name,
            "message": message,
        }
        log_data.update(extra)
        json_str = json.dumps(log_data)
        self.logger.error(json_str, exc_info=exc_info)
        print(json_str)

    def debug(self, message: str, extra: dict = None) -> None:
        """Log debug message with optional extra fields"""
        if extra is None:
            extra = {}
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "DEBUG",
            "logger": self.name,
            "message": message,
        }
        log_data.update(extra)
        json_str = json.dumps(log_data)
        self.logger.debug(json_str)
        print(json_str)


def get_logger(name: str) -> MyLogger:
    """Get a MyLogger instance"""
    return MyLogger(name)
        