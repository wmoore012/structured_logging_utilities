# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Wilton Moore

"""
Core structured logging functionality.

This module provides structured logging with JSON output and automatic performance timing,
extracted and enhanced from the Perday Catalog logging patterns.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class LogRecord(TypedDict, total=False):
    """Structured log record format."""

    timestamp: str
    level: str
    name: str
    message: str
    duration_ms: Optional[float]
    operation: Optional[str]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "message",
                }:
                    # Only include JSON-serializable values
                    try:
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        # Handle exceptions
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(log_data, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            # Fallback to string representation if JSON serialization fails
            fallback_data = {
                "timestamp": log_data["timestamp"],
                "level": log_data["level"],
                "name": log_data["name"],
                "message": f"JSON serialization failed: {e}",
                "original_message": str(record.getMessage()),
            }
            return json.dumps(fallback_data, ensure_ascii=False, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )


def get_logger(
    name: str,
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Get a structured logger with JSON formatting.

    Args:
        name: Logger name (typically module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO
        format_type: Format type ("json" or "text"). Defaults to LOG_FORMAT env var or "json"
        log_file: Optional file path for log output. Defaults to LOG_FILE env var

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("my_service")
        >>> logger.info("User login", extra={"user_id": 123, "ip": "192.168.1.1"})
        # Output: {"timestamp": "2024-01-01T12:00:00Z", "level": "INFO", "name": "my_service", "message": "User login", "user_id": 123, "ip": "192.168.1.1"}
    """
    # Get configuration from environment or defaults
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    format_type = format_type or os.getenv("LOG_FORMAT", "json").lower()
    log_file = log_file or os.getenv("LOG_FILE")

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set level
    try:
        logger.setLevel(getattr(logging, level))
    except AttributeError:
        logger.setLevel(logging.INFO)
        logger.warning(f"Invalid log level '{level}', using INFO")

    # Choose formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to create file handler for {log_file}: {e}")

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


@contextmanager
def log_performance(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    level: str = "INFO",
    threshold_ms: Optional[float] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for automatic performance timing.

    Args:
        operation: Name of the operation being timed
        context: Additional context to include in log
        logger: Logger instance to use. If None, creates one named "performance"
        level: Log level for the performance message
        threshold_ms: Only log if duration exceeds this threshold (milliseconds)

    Yields:
        Context dictionary that can be updated during operation

    Example:
        >>> with log_performance("database_query", {"table": "users", "query_type": "SELECT"}):
        ...     # Your database operation
        ...     results = db.query("SELECT * FROM users")
        # Automatically logs timing and context

        >>> # Update context during operation
        >>> with log_performance("bulk_insert") as ctx:
        ...     for i, record in enumerate(records):
        ...         insert_record(record)
        ...         if i % 1000 == 0:
        ...             ctx["processed"] = i
        ...     ctx["total_processed"] = len(records)
    """
    if logger is None:
        logger = get_logger("performance")

    # Initialize context
    operation_context = context.copy() if context else {}
    operation_context["operation"] = operation

    # Start timing
    start_time = time.perf_counter()
    start_timestamp = datetime.now(tz=timezone.utc).isoformat()

    try:
        # Yield context that can be updated during operation
        yield operation_context

        # Calculate duration
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Add timing information to context
        operation_context.update(
            {
                "duration_ms": round(duration_ms, 2),
                "start_time": start_timestamp,
                "end_time": datetime.now(tz=timezone.utc).isoformat(),
                "success": True,
            }
        )

        # Log if no threshold or duration exceeds threshold
        if threshold_ms is None or duration_ms >= threshold_ms:
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(
                log_level,
                f"Operation '{operation}' completed in {duration_ms:.2f}ms",
                extra=operation_context,
            )

    except Exception as e:
        # Calculate duration even on failure
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Add error information to context
        operation_context.update(
            {
                "duration_ms": round(duration_ms, 2),
                "start_time": start_timestamp,
                "end_time": datetime.now(tz=timezone.utc).isoformat(),
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )

        # Always log failures regardless of threshold
        logger.error(
            f"Operation '{operation}' failed after {duration_ms:.2f}ms: {e}",
            extra=operation_context,
        )

        # Re-raise the exception
        raise


def configure_root_logger(
    level: str = "INFO", format_type: str = "json", log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Configure the root logger with structured logging.

    This is useful for capturing logs from third-party libraries.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Format type ("json" or "text")
        log_file: Optional file path for log output

    Example:
        >>> configure_root_logger("DEBUG", "json", "app.log")
        >>> # Now all logging in the application uses structured JSON format
    """
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure like a regular logger
    logger = get_logger("root", level=level, format_type=format_type, log_file=log_file)

    # Copy handlers to root logger
    for handler in logger.handlers:
        root_logger.addHandler(handler)

    root_logger.setLevel(logger.level)


def create_child_logger(parent_logger: logging.Logger, child_name: str) -> logging.Logger:
    """
    Create a child logger that inherits parent configuration.

    Args:
        parent_logger: Parent logger instance
        child_name: Name for the child logger

    Returns:
        Child logger with inherited configuration

    Example:
        >>> main_logger = get_logger("myapp")
        >>> db_logger = create_child_logger(main_logger, "database")
        >>> api_logger = create_child_logger(main_logger, "api")
    """
    child_logger = logging.getLogger(f"{parent_logger.name}.{child_name}")

    # Copy handlers and level from parent
    for handler in parent_logger.handlers:
        child_logger.addHandler(handler)

    child_logger.setLevel(parent_logger.level)
    child_logger.propagate = False

    return child_logger


# Convenience function for backward compatibility with existing Perday Catalog code
def setup_logging(name: str = "icatalog") -> logging.Logger:
    """
    Legacy setup function for backward compatibility.

    This maintains compatibility with existing Perday Catalog logging setup
    while providing enhanced structured logging capabilities.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return get_logger(name)
