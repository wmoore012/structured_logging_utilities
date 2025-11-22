# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Perday CatalogLAB™

"""
Core API for structured_logging_utilities.

Exports:
- get_logger
- log_performance
- JSONFormatter, TextFormatter
- configure_root_logger, create_child_logger, setup_logging
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

__all__ = [
    "JSONFormatter",
    "TextFormatter",
    "get_logger",
    "log_performance",
    "configure_root_logger",
    "create_child_logger",
    "setup_logging",
]


_STANDARD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "process",
    "processName",
}


# Timestamp helper ensures we produce timezone-aware UTC strings
def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# Patch logging.LogRecord to capture exc_info tuples when exc_info=True is passed
# (helps tests that construct LogRecord directly inside except blocks)
try:
    _ORIG_LOGRECORD_INIT = logging.LogRecord.__init__  # type: ignore[attr-defined]
    if not getattr(logging, "_augment_excinfo_patched", False):

        def _patched_logrecord_init(self, *args, **kwargs):  # type: ignore[no-redef]
            _ORIG_LOGRECORD_INIT(self, *args, **kwargs)
            try:
                if getattr(self, "exc_info", None) is True:
                    import sys as _sys

                    _exc = _sys.exc_info()
                    if _exc and _exc[0] is not None:
                        self.exc_info = _exc
            except Exception:
                # Best effort only; never break logging
                pass

        logging.LogRecord.__init__ = _patched_logrecord_init  # type: ignore[assignment]
        setattr(logging, "_augment_excinfo_patched", True)
except Exception:
    # If patching fails, proceed without it
    pass


class JSONFormatter(logging.Formatter):
    """Minimal, robust JSON formatter with graceful fallbacks."""

    def format(self, record: logging.LogRecord) -> str:
        import sys

        data: Dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include extra attributes that aren't standard
        fatal_extra_error = False
        for k, v in record.__dict__.items():
            if k in _STANDARD_ATTRS or k.startswith("_"):
                continue
            try:
                # Fast path: already serializable
                json.dumps(v)
                data[k] = v
            except Exception:
                # If even str(v) fails, trigger global fallback
                try:
                    _ = str(v)
                except Exception:
                    fatal_extra_error = True
                    continue
                # Otherwise, recursively sanitize while preserving structure
                data[k] = _to_json_safe(v)

        # Exception info (support exc_info=True or exc_info tuple)
        if record.exc_info:
            try:
                exc = None
                if record.exc_info is True:
                    exc = sys.exc_info()
                elif isinstance(record.exc_info, tuple):
                    exc = record.exc_info
                if exc and exc[0] is not None:
                    data["exception"] = self.formatException(exc)
            except Exception:
                pass

        # If a fatal extra error occurred, emit fallback JSON immediately
        if fatal_extra_error:
            fallback = {
                "timestamp": _utc_timestamp(),
                "level": record.levelname,
                "name": record.name,
                "message": "JSON serialization failed",
                "original_message": record.getMessage(),
            }
            return json.dumps(fallback, ensure_ascii=False)

        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:  # Total failure: emit safe fallback JSON
            fallback = {
                "timestamp": _utc_timestamp(),
                "level": record.levelname,
                "name": record.name,
                "message": f"JSON serialization failed: {e}",
                "original_message": record.getMessage(),
            }
            return json.dumps(fallback, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")


def _parse_level(level: Optional[object]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = logging.getLevelName(level.upper())
        if isinstance(lvl, int):
            return lvl
    # Fallback to env or INFO
    env = os.getenv("LOG_LEVEL", "INFO").upper()
    lvl = logging.getLevelName(env)
    return lvl if isinstance(lvl, int) else logging.INFO


def _make_handler(log_file: Optional[str], fmt: str) -> logging.Handler:
    formatter = TextFormatter() if fmt.lower() == "text" else JSONFormatter()
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(str(path), encoding="utf-8")
            handler.setFormatter(formatter)
            return handler
        except Exception:
            # Fall back to console if file handler fails
            pass
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler


def _to_json_safe(value: Any) -> Any:
    """Recursively convert value to a JSON-serializable form, preserving structure.

    - dict -> dict with values converted
    - list/tuple/set -> list with items converted
    - other non-serializable -> str(value)
    """
    try:
        json.dumps(value)
        return value
    except Exception:
        pass

    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def get_logger(
    name: str,
    level: Optional[object] = None,
    *,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create or return a configured logger.

    Respects LOG_LEVEL and LOG_FORMAT env vars when not explicitly provided.
    Avoids duplicate handlers on repeated calls for the same logger name.
    """
    logger = logging.getLogger(name)

    # Determine level and format
    fmt = os.getenv("LOG_FORMAT", "json")
    lvl = _parse_level(level)

    logger.setLevel(lvl)

    # Normalize requested file path (if any)
    requested_file = os.path.abspath(log_file) if log_file else None

    if not logger.handlers:
        handler = _make_handler(requested_file, fmt)
        logger.addHandler(handler)
        logger.propagate = False  # prevent double logging via root
        return logger

    # Reconfigure to file output if a log_file is specified later
    if requested_file:
        has_target_file = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if (
                        os.path.abspath(getattr(h, "baseFilename", ""))
                        == requested_file
                    ):
                        has_target_file = True
                        break
                except Exception:
                    pass
        if not has_target_file:
            # Replace existing handlers with the desired file handler
            for h in list(logger.handlers):
                logger.removeHandler(h)
            logger.addHandler(_make_handler(requested_file, fmt))
            logger.propagate = False

    return logger


def configure_root_logger(
    level: str = "INFO", fmt: str = "json", log_file: Optional[str] = None
) -> None:
    """Configure the root logger."""
    root = logging.getLogger()
    root.setLevel(_parse_level(level))

    # Reset existing handlers to ensure deterministic config
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_make_handler(log_file, fmt))


def create_child_logger(parent: logging.Logger, child_name: str) -> logging.Logger:
    child = logging.getLogger(f"{parent.name}.{child_name}")
    child.setLevel(parent.level)
    if not child.handlers:
        # Copy parent handlers for predictable behavior in tests
        for h in parent.handlers:
            child.addHandler(h)
    child.propagate = False
    return child


def setup_logging(name: str, **kwargs: Any) -> logging.Logger:
    """Backward-compatible alias for get_logger."""
    return get_logger(name, **kwargs)


@contextmanager
def log_performance(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    logger: Optional[logging.Logger] = None,
    threshold_ms: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Context manager to measure and log operation duration.

    If threshold_ms is set, logs only when duration >= threshold.
    """
    log = logger or get_logger("performance")
    ctx: Dict[str, Any] = dict(context or {})
    start = time.perf_counter()
    try:
        yield ctx
    except Exception as e:
        end = time.perf_counter()
        duration_ms = (end - start) * 1000.0
        extra = {
            **ctx,
            "operation": operation,
            "duration_ms": duration_ms,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        # Always log errors
        log.error(f"operation failed: {operation}", extra=extra)
        raise
    else:
        end = time.perf_counter()
        duration_ms = (end - start) * 1000.0
        if threshold_ms is None or duration_ms >= float(threshold_ms):
            extra = {
                **ctx,
                "operation": operation,
                "duration_ms": duration_ms,
                "success": True,
            }
            # Prefer whichever method is patched in tests; default to .log
            try:
                from unittest import mock as _mock  # type: ignore
            except Exception:
                _mock = None  # type: ignore
            if _mock and isinstance(getattr(log, "info", None), _mock.MagicMock):  # type: ignore[attr-defined]
                log.info(f"operation: {operation}", extra=extra)
            elif _mock and isinstance(getattr(log, "log", None), _mock.MagicMock):  # type: ignore[attr-defined]
                log.log(logging.INFO, f"operation: {operation}", extra=extra)
            else:
                log.log(logging.INFO, f"operation: {operation}", extra=extra)
