"""
Logger Module

This module provides custom logging functionality for the causal-graph-llm-evaluator project.
It includes:
- Colored console output with custom formatting
- File logging with both general and error-specific log files
- Configuration options for log levels and output destinations
- Automatic log file rotation with date-based filenames

Usage:
    from logger import get_logger

    # Get a logger instance
    logger = get_logger(filename=__file__, console_color="green")

    # Use the logger
    logger.info("Processing started")
    logger.warning("Resource usage high")
    logger.error("An error occurred")
"""

import logging
import os
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to console log messages.
    """

    # ANSI color codes
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'reset': '\033[0m'
    }

    # Default level colors (can be overridden)
    LEVEL_COLORS = {
        'DEBUG': 'bright_black',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'black'
    }

    def __init__(self, fmt=None, datefmt=None, color='white'):
        super().__init__(fmt, datefmt)
        self.base_color = color
        # Make INFO level use the specified console color
        self.LEVEL_COLORS = self.LEVEL_COLORS.copy()  # Create a copy to avoid modifying the class attribute
        self.LEVEL_COLORS['INFO'] = color

    def format(self, record):
        # Get the formatted message
        formatted = super().format(record)

        # Choose color based on level or use base color
        if record.levelname in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelname]
        else:
            color = self.base_color

        # Apply color if available
        if color in self.COLORS:
            colored_message = f"{self.COLORS[color]}{formatted}{self.COLORS['reset']}"
            return colored_message

        return formatted


def get_logger(filename=__file__, level: str = "INFO", console_color: str = "blue", custom_name: str = None, function_name: bool = False) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        filename (str): Path to the file, typically __file__ from the calling module
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_color (str): Default color for console output. Available colors:
                           'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
                           'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                           'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        custom_name (str): Optional custom name for the logger instead of the filename
        function_name (bool): Whether to include function name in log messages (default: True)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Extract basename from filename or use custom_name if provided
    name = custom_name if custom_name else os.path.basename(filename)

    # Create logger
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Create formatters - with or without function name based on parameter

    detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


    # Create colored formatter for console
    if function_name:
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            color=console_color
        )
    else:
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            color=console_color
        )

    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)  # Use the same level as the logger
    console_handler.setFormatter(colored_formatter)

    # Create file handler for all logs
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, f"app_{current_date}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Create error file handler
    error_handler = logging.FileHandler(
        os.path.join(logs_dir, f"errors_{current_date}.log"),
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Update the logging level for a logger.

    Args:
        logger (logging.Logger): Logger instance to update
        level (str): New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


# Default logger instance
default_logger = get_logger(custom_name="logger", console_color="blue")

if __name__ == '__main__':
    default_logger.info("Info Message")
    default_logger.warning("Warning Message")
    default_logger.error("Error Message")
    default_logger.critical("Critical Message")
    default_logger.debug("Debug Message")
