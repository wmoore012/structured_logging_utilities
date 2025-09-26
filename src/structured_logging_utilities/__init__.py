# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Wilton Moore

"""
Structured Logging Utilities

Production-ready structured logging with JSON output and automatic performance timing.
"""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    JSONFormatter,
    TextFormatter,
    configure_root_logger,
    create_child_logger,
    get_logger,
    log_performance,
    setup_logging,
)

try:
    __version__ = version("structured-logging-utilities")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Alias for common usage
setup_logger = setup_logging
StructuredFormatter = JSONFormatter

__all__ = [
    "get_logger",
    "setup_logger",
    "setup_logging",
    "log_performance",
    "JSONFormatter",
    "StructuredFormatter",
    "TextFormatter",
    "configure_root_logger",
    "create_child_logger",
]
