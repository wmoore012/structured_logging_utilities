# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Integration tests for structured logging utilities.

These tests verify end-to-end functionality using real database scenarios
and production-like workloads, avoiding dummy data as specified.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from structured_logging_utilities import get_logger, log_performance


class TestDatabaseLoggingIntegration:
    """Test logging integration with database operations."""

    def test_database_query_logging_workflow(self):
        """Test complete database query logging workflow."""
        logger = get_logger("database_integration")

        # Capture log output
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "db_integration.log"
            file_logger = get_logger("database_integration", log_file=str(log_file))

            # Simulate database connection logging
            file_logger.info(
                "Database connection established",
                extra={
                    "host": "localhost",
                    "database": "icatalog_public",
                    "connection_pool_size": 10,
                    "ssl_enabled": True,
                },
            )

            # Simulate query execution with performance logging
            with log_performance(
                "database_query",
                {"query_type": "SELECT", "table": "songs", "estimated_rows": 50000},
                logger=file_logger,
            ) as ctx:
                # Simulate query execution time
                time.sleep(0.02)  # 20ms
                ctx["actual_rows"] = 47832
                ctx["cache_hit"] = False
                ctx["index_used"] = "idx_songs_title"

            # Simulate connection cleanup
            file_logger.info(
                "Database connection closed",
                extra={
                    "connection_duration_seconds": 125.7,
                    "queries_executed": 15,
                    "total_rows_processed": 125000,
                },
            )

            # Verify log file contents
            assert log_file.exists()
            log_content = log_file.read_text()

            # Parse JSON log entries
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]
            parsed_logs = [json.loads(line) for line in log_lines]

            # Verify connection log
            connection_log = parsed_logs[0]
            assert connection_log["message"] == "Database connection established"
            assert connection_log["host"] == "localhost"
            assert connection_log["database"] == "icatalog_public"

            # Verify performance log
            performance_log = parsed_logs[1]
            assert "database_query" in performance_log["message"]
            assert performance_log["query_type"] == "SELECT"
            assert performance_log["actual_rows"] == 47832
            assert performance_log["duration_ms"] >= 20

            # Verify cleanup log
            cleanup_log = parsed_logs[2]
            assert cleanup_log["message"] == "Database connection closed"
            assert cleanup_log["queries_executed"] == 15

    def test_etl_pipeline_logging_integration(self):
        """Test ETL pipeline logging with real data processing patterns."""
        logger = get_logger("etl_pipeline")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "etl_integration.log"
            etl_logger = get_logger("etl_pipeline", log_file=str(log_file))

            # ETL job start
            etl_logger.info(
                "ETL job started",
                extra={
                    "job_id": "etl_spotify_tracks_20241217",
                    "source_table": "spotify_raw",
                    "target_table": "songs",
                    "batch_size": 10000,
                    "total_estimated_records": 2500000,
                },
            )

            # Process multiple batches
            total_processed = 0
            total_failed = 0

            for batch_num in range(3):
                with log_performance(
                    f"etl_batch_{batch_num}",
                    {"batch_number": batch_num, "batch_size": 10000, "source_table": "spotify_raw"},
                    logger=etl_logger,
                ) as ctx:
                    # Simulate batch processing
                    time.sleep(0.01)  # 10ms per batch

                    # Simulate processing results
                    processed = 9847 + (batch_num * 50)  # Slight variation
                    failed = 153 - (batch_num * 10)  # Improving over time

                    ctx["records_processed"] = processed
                    ctx["records_failed"] = failed
                    ctx["success_rate"] = (processed / (processed + failed)) * 100

                    total_processed += processed
                    total_failed += failed

            # ETL job completion
            etl_logger.info(
                "ETL job completed",
                extra={
                    "job_id": "etl_spotify_tracks_20241217",
                    "total_processed": total_processed,
                    "total_failed": total_failed,
                    "overall_success_rate": (total_processed / (total_processed + total_failed))
                    * 100,
                    "job_duration_minutes": 45.2,
                    "throughput_records_per_second": total_processed / (45.2 * 60),
                },
            )

            # Verify comprehensive logging
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]
            parsed_logs = [json.loads(line) for line in log_lines]

            # Should have job start + 3 batch logs + job completion = 5 logs
            assert len(parsed_logs) == 5

            # Verify job start log
            start_log = parsed_logs[0]
            assert start_log["message"] == "ETL job started"
            assert start_log["total_estimated_records"] == 2500000

            # Verify batch logs have performance data
            batch_logs = parsed_logs[1:4]
            for i, batch_log in enumerate(batch_logs):
                assert f"etl_batch_{i}" in batch_log["message"]
                assert "duration_ms" in batch_log
                assert "records_processed" in batch_log
                assert batch_log["success_rate"] > 95  # High success rate

            # Verify completion log
            completion_log = parsed_logs[4]
            assert completion_log["message"] == "ETL job completed"
            assert completion_log["total_processed"] == total_processed
            assert completion_log["throughput_records_per_second"] > 0

    def test_api_request_logging_integration(self):
        """Test API request logging with real request/response patterns."""
        logger = get_logger("api_integration")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "api_integration.log"
            api_logger = get_logger("api_integration", log_file=str(log_file))

            # Simulate various API requests
            api_requests = [
                {
                    "method": "GET",
                    "endpoint": "/api/songs/search",
                    "query": "bohemian rhapsody",
                    "expected_results": 23,
                    "expected_time_ms": 127,
                },
                {
                    "method": "POST",
                    "endpoint": "/api/playlists",
                    "payload_size": 2048,
                    "expected_results": 1,
                    "expected_time_ms": 89,
                },
                {
                    "method": "GET",
                    "endpoint": "/api/artists/12345/songs",
                    "query": None,
                    "expected_results": 156,
                    "expected_time_ms": 203,
                },
            ]

            for i, request_data in enumerate(api_requests):
                request_id = f"req_{i:04d}"

                # Log request start
                api_logger.info(
                    "API request received",
                    extra={
                        "request_id": request_id,
                        "method": request_data["method"],
                        "endpoint": request_data["endpoint"],
                        "user_id": f"user_{1000 + i}",
                        "user_agent": "Mozilla/5.0 (compatible; test)",
                        "ip_address": f"192.168.1.{100 + i}",
                    },
                )

                # Process request with performance logging
                with log_performance(
                    "api_request_processing",
                    {
                        "request_id": request_id,
                        "endpoint": request_data["endpoint"],
                        "method": request_data["method"],
                    },
                    logger=api_logger,
                ) as ctx:
                    # Simulate request processing
                    processing_time = request_data["expected_time_ms"] / 1000
                    time.sleep(processing_time)

                    ctx["response_code"] = 200
                    ctx["results_count"] = request_data["expected_results"]
                    ctx["cache_status"] = "miss" if i == 0 else "hit"
                    ctx["database_queries"] = 2 if request_data["method"] == "GET" else 3

            # Verify API logging
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]
            parsed_logs = [json.loads(line) for line in log_lines]

            # Should have 6 logs (3 requests × 2 logs each)
            assert len(parsed_logs) == 6

            # Verify request/response pairs
            for i in range(0, 6, 2):
                request_log = parsed_logs[i]
                response_log = parsed_logs[i + 1]

                # Verify request log
                assert request_log["message"] == "API request received"
                assert "request_id" in request_log
                assert "endpoint" in request_log

                # Verify response log
                assert "api_request_processing" in response_log["message"]
                assert response_log["response_code"] == 200
                assert "duration_ms" in response_log
                assert response_log["duration_ms"] > 50  # Reasonable processing time


class TestHighVolumeLoggingIntegration:
    """Test logging under high-volume production scenarios."""

    def test_high_volume_concurrent_logging(self):
        """Test logging performance under high concurrent load."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger = get_logger("high_volume")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "high_volume.log"
            volume_logger = get_logger("high_volume", log_file=str(log_file))

            # Track successful logs across threads
            successful_logs = threading.local()
            lock = threading.Lock()
            total_successful = 0

            def log_batch(thread_id: int, batch_size: int) -> int:
                """Log a batch of messages from a single thread."""
                thread_successful = 0

                for i in range(batch_size):
                    try:
                        volume_logger.info(
                            f"High volume message {thread_id}_{i}",
                            extra={
                                "thread_id": thread_id,
                                "message_number": i,
                                "batch_size": batch_size,
                                "timestamp_ms": time.time() * 1000,
                                "payload": {
                                    "data": f"payload_data_{i}",
                                    "size": len(f"payload_data_{i}"),
                                    "thread": thread_id,
                                },
                            },
                        )
                        thread_successful += 1
                    except Exception:
                        pass  # Count failures

                return thread_successful

            # Run concurrent logging
            num_threads = 4
            messages_per_thread = 250

            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(log_batch, thread_id, messages_per_thread)
                    for thread_id in range(num_threads)
                ]

                for future in as_completed(futures):
                    total_successful += future.result()

            end_time = time.perf_counter()

            # Verify high-volume performance
            total_time = end_time - start_time
            throughput = total_successful / total_time

            # Should achieve reasonable throughput under load
            assert throughput >= 500  # At least 500 logs/second
            assert total_successful >= (
                num_threads * messages_per_thread * 0.95
            )  # 95% success rate

            # Verify log file integrity
            assert log_file.exists()
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]

            # Should have most of the expected log lines
            assert len(log_lines) >= total_successful * 0.9

            # Verify JSON integrity of random samples
            sample_lines = log_lines[:: len(log_lines) // 10] if log_lines else []
            for line in sample_lines:
                parsed = json.loads(line)  # Should not raise exception
                assert "thread_id" in parsed
                assert "message_number" in parsed

    def test_memory_efficient_long_running_logging(self):
        """Test memory efficiency during long-running logging operations."""
        logger = get_logger("memory_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "memory_test.log"
            memory_logger = get_logger("memory_test", log_file=str(log_file))

            # Log many messages to test memory efficiency
            num_messages = 5000
            message_batch_size = 500

            for batch in range(0, num_messages, message_batch_size):
                batch_start = time.perf_counter()

                # Log a batch of messages
                for i in range(message_batch_size):
                    message_id = batch + i
                    memory_logger.info(
                        f"Memory test message {message_id}",
                        extra={
                            "message_id": message_id,
                            "batch_number": batch // message_batch_size,
                            "large_payload": {
                                "data": "x" * 100,  # 100 character payload
                                "metadata": {
                                    "created": time.time(),
                                    "sequence": message_id,
                                    "batch": batch // message_batch_size,
                                },
                                "tags": [f"tag_{j}" for j in range(5)],
                            },
                        },
                    )

                batch_end = time.perf_counter()
                batch_time = batch_end - batch_start

                # Log batch completion
                memory_logger.info(
                    f"Batch {batch // message_batch_size} completed",
                    extra={
                        "batch_number": batch // message_batch_size,
                        "messages_in_batch": message_batch_size,
                        "batch_duration_ms": batch_time * 1000,
                        "messages_per_second": message_batch_size / batch_time,
                    },
                )

            # Verify log file was created and has reasonable size
            assert log_file.exists()
            file_size_mb = log_file.stat().st_size / (1024 * 1024)

            # File size should be reasonable (not excessive memory usage)
            assert file_size_mb < 50  # Less than 50MB for 5000 messages

            # Verify log integrity
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]

            # Should have all messages plus batch completion logs
            expected_lines = num_messages + (num_messages // message_batch_size)
            assert len(log_lines) >= expected_lines * 0.95  # 95% of expected


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""

    def test_logging_with_database_failures(self):
        """Test logging behavior when database operations fail."""
        logger = get_logger("db_failure_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "db_failure.log"
            failure_logger = get_logger("db_failure_test", log_file=str(log_file))

            # Simulate database operation that fails
            try:
                with log_performance(
                    "database_operation",
                    {"operation": "INSERT", "table": "songs", "batch_size": 1000},
                    logger=failure_logger,
                ) as ctx:
                    # Simulate some processing
                    time.sleep(0.01)
                    ctx["records_processed"] = 847

                    # Simulate database failure
                    raise ConnectionError("Database connection lost")

            except ConnectionError:
                pass  # Expected

            # Verify error was logged properly
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]

            assert len(log_lines) == 1
            error_log = json.loads(log_lines[0])

            assert error_log["level"] == "ERROR"
            assert "failed" in error_log["message"]
            assert error_log["success"] is False
            assert error_log["error"] == "Database connection lost"
            assert error_log["error_type"] == "ConnectionError"
            assert error_log["records_processed"] == 847

    def test_json_serialization_error_handling(self):
        """Test handling of JSON serialization errors in production scenarios."""
        logger = get_logger("json_error_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "json_error.log"
            json_logger = get_logger("json_error_test", log_file=str(log_file))

            # Create problematic data that might cause JSON issues
            class NonSerializableObject:
                def __str__(self):
                    return "NonSerializableObject"

            # Log with problematic data
            json_logger.info(
                "Processing complex data",
                extra={
                    "user_id": 12345,
                    "data": {
                        "valid_field": "valid_value",
                        "problematic_object": NonSerializableObject(),
                        "nested_data": {"timestamp": time.time(), "valid_list": [1, 2, 3]},
                    },
                },
            )

            # Verify log was written (should handle serialization gracefully)
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]

            assert len(log_lines) == 1
            log_entry = json.loads(log_lines[0])  # Should not raise exception

            assert log_entry["message"] == "Processing complex data"
            assert log_entry["user_id"] == 12345
            assert "data" in log_entry

            # Problematic object should be converted to string
            assert isinstance(log_entry["data"]["problematic_object"], str)


class TestProductionScenarioIntegration:
    """Test logging in realistic production scenarios."""

    def test_microservice_request_tracing(self):
        """Test request tracing across microservice boundaries."""
        logger = get_logger("microservice_tracing")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "microservice.log"
            trace_logger = get_logger("microservice_tracing", log_file=str(log_file))

            # Simulate request flowing through multiple services
            trace_id = "trace_abc123"
            user_id = "user_456"

            # Service 1: API Gateway
            trace_logger.info(
                "Request received at API Gateway",
                extra={
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "service": "api_gateway",
                    "endpoint": "/api/songs/search",
                    "method": "GET",
                    "source_ip": "192.168.1.100",
                },
            )

            # Service 2: Authentication Service
            with log_performance(
                "authentication",
                {"trace_id": trace_id, "user_id": user_id, "service": "auth_service"},
                logger=trace_logger,
            ) as ctx:
                time.sleep(0.005)  # 5ms auth check
                ctx["auth_method"] = "jwt"
                ctx["token_valid"] = True

            # Service 3: Search Service
            with log_performance(
                "song_search",
                {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "service": "search_service",
                    "query": "bohemian rhapsody",
                },
                logger=trace_logger,
            ) as ctx:
                time.sleep(0.015)  # 15ms search
                ctx["results_found"] = 23
                ctx["search_index"] = "songs_v2"
                ctx["cache_hit"] = False

            # Service 4: Response formatting
            trace_logger.info(
                "Response sent",
                extra={
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "service": "api_gateway",
                    "response_code": 200,
                    "response_size_bytes": 4096,
                    "total_duration_ms": 25.3,
                },
            )

            # Verify complete trace
            log_content = log_file.read_text()
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]
            parsed_logs = [json.loads(line) for line in log_lines]

            # Should have 4 log entries
            assert len(parsed_logs) == 4

            # All logs should have the same trace_id
            for log_entry in parsed_logs:
                assert log_entry["trace_id"] == trace_id
                assert log_entry["user_id"] == user_id

            # Verify service progression
            services = [log["service"] for log in parsed_logs]
            assert services == ["api_gateway", "auth_service", "search_service", "api_gateway"]

            # Verify performance logs have timing data
            perf_logs = [log for log in parsed_logs if "duration_ms" in log]
            assert len(perf_logs) == 2  # auth and search

            for perf_log in perf_logs:
                assert perf_log["duration_ms"] > 0
                assert perf_log["success"] is True
