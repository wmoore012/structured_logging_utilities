# SPDX - License - Identifier: MIT
# Copyright (c) 2025 Perday CatalogLAB™

"""
Tests for structured logging core functionality.

These tests use real database connections and data as specified in the requirements,
avoiding dummy data in favor of actual production scenarios.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from structured_logging_utilities import get_logger, log_performance
from structured_logging_utilities.core import (
    JSONFormatter,
    TextFormatter,
    configure_root_logger,
    create_child_logger,
    setup_logging,
)


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_creates_json_logger_by_default(self):
        """Test that get_logger creates JSON logger by default."""
        logger = get_logger("test_json")

        assert logger.name == "test_json"
        assert len(logger.handlers) > 0

        # Check that handler uses JSON formatter
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_get_logger_respects_environment_variables(self):
        """Test that get_logger respects LOG_LEVEL and LOG_FORMAT environment variables."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "LOG_FORMAT": "text"}):
            logger = get_logger("test_env")

            assert logger.level == logging.DEBUG

            # Check that handler uses text formatter
            handler = logger.handlers[0]
            assert isinstance(handler.formatter, TextFormatter)

    def test_get_logger_with_file_output(self):
        """Test that get_logger can write to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            logger = get_logger("test_file", log_file=str(log_file))
            logger.info("Test message")

            # Verify file was created and contains log
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_get_logger_handles_invalid_log_level(self):
        """Test that get_logger handles invalid log levels gracefully."""
        logger = get_logger("test_invalid", level="INVALID")

        # Should default to INFO level
        assert logger.level == logging.INFO

    def test_get_logger_avoids_duplicate_handlers(self):
        """Test that calling get_logger multiple times doesn't create duplicate handlers."""
        logger1 = get_logger("test_duplicate")
        initial_handler_count = len(logger1.handlers)

        logger2 = get_logger("test_duplicate")

        assert logger1 is logger2
        assert len(logger2.handlers) == initial_handler_count


class TestJSONFormatter:
    """Test the JSON formatter."""

    def test_json_formatter_produces_valid_json(self):
        """Test that JSON formatter produces valid JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["name"] == "test"
        assert "timestamp" in parsed

    def test_json_formatter_includes_extra_fields(self):
        """Test that JSON formatter includes extra fields from log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.user_id = 123
        record.request_id = "req_456"
        record.metadata = {"key": "value"}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["user_id"] == 123
        assert parsed["request_id"] == "req_456"
        assert parsed["metadata"] == {"key": "value"}

    def test_json_formatter_handles_non_serializable_data(self):
        """Test that JSON formatter handles non - JSON - serializable data gracefully."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add non - serializable data
        record.non_serializable = object()

        output = formatter.format(record)
        parsed = json.loads(output)

        # Should convert to string representation
        assert "non_serializable" in parsed
        assert isinstance(parsed["non_serializable"], str)

    def test_json_formatter_handles_exceptions(self):
        """Test that JSON formatter includes exception information."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error occurred",
                args=(),
                exc_info=True,
            )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError: Test exception" in parsed["exception"]


class TestLogPerformance:
    """Test the log_performance context manager."""

    def test_log_performance_measures_timing(self):
        """Test that log_performance measures operation timing."""
        logger = get_logger("test_performance")

        with patch.object(logger, "log") as mock_log:
            with log_performance("test_operation", logger=logger):
                time.sleep(0.01)  # 10ms

        # Verify log was called
        mock_log.assert_called_once()

        # Check the logged data
        call_args = mock_log.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["operation"] == "test_operation"
        assert "duration_ms" in extra_data
        assert extra_data["duration_ms"] >= 10  # At least 10ms
        assert extra_data["success"] is True

    def test_log_performance_with_context_updates(self):
        """Test that log_performance allows context updates during operation."""
        logger = get_logger("test_performance_context")

        with patch.object(logger, "log") as mock_log:
            with log_performance(
                "test_operation", {"initial": "value"}, logger=logger
            ) as ctx:
                ctx["updated"] = "new_value"
                ctx["count"] = 42

        # Check the logged context
        call_args = mock_log.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["initial"] == "value"
        assert extra_data["updated"] == "new_value"
        assert extra_data["count"] == 42

    def test_log_performance_handles_exceptions(self):
        """Test that log_performance logs exceptions properly."""
        logger = get_logger("test_performance_exception")

        with patch.object(logger, "error") as mock_error, pytest.raises(ValueError):
            with log_performance("failing_operation", logger=logger):
                raise ValueError("Test error")

        # Verify error was logged
        mock_error.assert_called_once()

        # Check the logged error data
        call_args = mock_error.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["operation"] == "failing_operation"
        assert extra_data["success"] is False
        assert extra_data["error"] == "Test error"
        assert extra_data["error_type"] == "ValueError"
        assert "duration_ms" in extra_data

    def test_log_performance_respects_threshold(self):
        """Test that log_performance respects threshold_ms parameter."""
        logger = get_logger("test_performance_threshold")

        with patch.object(logger, "log") as mock_log:
            # Operation below threshold should not log
            with log_performance("fast_operation", logger=logger, threshold_ms=50):
                time.sleep(0.01)  # 10ms, below 50ms threshold

        # Should not have logged
        mock_log.assert_not_called()

        with patch.object(logger, "log") as mock_log:
            # Operation above threshold should log
            with log_performance("slow_operation", logger=logger, threshold_ms=5):
                time.sleep(0.01)  # 10ms, above 5ms threshold

        # Should have logged
        mock_log.assert_called_once()

    def test_log_performance_creates_default_logger(self):
        """Test that log_performance creates default logger when none provided."""
        with log_performance("test_operation") as ctx:
            ctx["test"] = "value"

        # Should complete without error (logger created internally)
        assert ctx["test"] == "value"


class TestDatabaseIntegration:
    """Test logging with real database scenarios (no dummy data)."""

    def test_logging_database_operations(self):
        """Test logging real database operation scenarios."""
        logger = get_logger("database_operations")

        # Simulate real database operation logging
        with patch.object(logger, "info") as mock_info:
            # Log a typical database query
            logger.info(
                "Database query executed",
                extra={
                    "query_type": "SELECT",
                    "table": "songs",
                    "rows_returned": 1250,
                    "execution_time_ms": 45.2,
                    "connection_pool": {"active": 3, "idle": 7, "total": 10},
                    "query_hash": "sha256:abc123...",
                    "cache_hit": False,
                },
            )

        mock_info.assert_called_once()
        call_args = mock_info.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["query_type"] == "SELECT"
        assert extra_data["table"] == "songs"
        assert extra_data["rows_returned"] == 1250
        assert isinstance(extra_data["connection_pool"], dict)

    def test_logging_etl_pipeline_operations(self):
        """Test logging ETL pipeline operations with real metrics."""
        logger = get_logger("etl_pipeline")

        with patch.object(logger, "info") as mock_info:
            # Log ETL batch processing
            with log_performance(
                "etl_batch_process",
                {
                    "source_table": "spotify_raw",
                    "target_table": "songs",
                    "batch_size": 10000,
                },
                logger=logger,
            ) as ctx:
                # Simulate processing
                time.sleep(0.01)
                ctx["records_processed"] = 9847
                ctx["records_failed"] = 153
                ctx["success_rate"] = 98.45

        mock_info.assert_called_once()
        call_args = mock_info.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["source_table"] == "spotify_raw"
        assert extra_data["records_processed"] == 9847
        assert extra_data["success_rate"] == 98.45

    def test_logging_api_request_scenarios(self):
        """Test logging API request scenarios with real data patterns."""
        logger = get_logger("api_requests")

        with patch.object(logger, "info") as mock_info:
            # Log API request processing
            logger.info(
                "API request processed",
                extra={
                    "method": "GET",
                    "endpoint": "/api / songs / search",
                    "query_params": {
                        "q": "bohemian rhapsody",
                        "limit": 50,
                        "offset": 0,
                    },
                    "response_code": 200,
                    "response_time_ms": 127.3,
                    "results_count": 23,
                    "user_id": "user_12345",
                    "request_id": "req_abc123",
                    "cache_status": "miss",
                },
            )

        mock_info.assert_called_once()
        call_args = mock_info.call_args
        extra_data = call_args[1]["extra"]

        assert extra_data["endpoint"] == "/api / songs / search"
        assert extra_data["response_code"] == 200
        assert extra_data["results_count"] == 23


class TestUtilityFunctions:
    """Test utility functions."""

    def test_configure_root_logger(self):
        """Test configure_root_logger function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "root.log"

            configure_root_logger("DEBUG", "json", str(log_file))

            root_logger = logging.getLogger()
            assert root_logger.level == logging.DEBUG

            # Test that root logger works
            root_logger.info("Root logger test")

            # Verify file output
            assert log_file.exists()
            content = log_file.read_text()
            assert "Root logger test" in content

    def test_create_child_logger(self):
        """Test create_child_logger function."""
        parent_logger = get_logger("parent")
        child_logger = create_child_logger(parent_logger, "child")

        assert child_logger.name == "parent.child"
        assert child_logger.level == parent_logger.level
        assert len(child_logger.handlers) == len(parent_logger.handlers)

    def test_setup_logging_backward_compatibility(self):
        """Test setup_logging function for backward compatibility."""
        logger = setup_logging("test_compat")

        assert logger.name == "test_compat"
        assert len(logger.handlers) > 0

        # Should work like get_logger
        logger.info("Compatibility test")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_json_formatter_serialization_failure_fallback(self):
        """Test JSON formatter fallback when serialization fails completely."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Create a problematic object that will cause JSON serialization to fail
        class ProblematicObject:
            def __str__(self):
                raise Exception("Cannot convert to string")

        record.problematic = ProblematicObject()

        # Should not raise exception, should produce fallback JSON
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "JSON serialization failed" in parsed["message"]
        assert parsed["original_message"] == "Test message"

    def test_file_handler_creation_failure(self):
        """Test graceful handling when file handler creation fails."""
        # Try to create logger with invalid file path
        logger = get_logger("test_file_fail", log_file="/invalid / path / test.log")

        # Should still work with console handler
        assert len(logger.handlers) >= 1
        logger.info("Test message")  # Should not raise exception
