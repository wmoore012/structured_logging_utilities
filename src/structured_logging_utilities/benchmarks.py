# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Perday CatalogLAB™

"""
Benchmarking suite for structured logging utilities performance.

This module provides comprehensive benchmarking capabilities to measure
logging overhead, JSON serialization performance, and I / O throughput under high load.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from math import ceil

try:
    import psutil
except ImportError:
    psutil = None

from .core import get_logger, log_performance


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""

    test_name: str
    operations_completed: int
    time_seconds: float
    memory_mb: float
    ops_per_second: int
    throughput_mb_per_second: float
    success_rate: float
    metadata: dict[str, Any]


def benchmark_logging_throughput(
    log_count: int = 10000, concurrent_threads: int = 1
) -> BenchmarkResult:
    """
    Benchmark logging throughput under various loads.

    Args:
        log_count: Number of log messages to generate
        concurrent_threads: Number of concurrent threads for stress testing

    Returns:
        BenchmarkResult with throughput metrics

    Example:
        >>> result = benchmark_logging_throughput(10000, 4)
        >>> print(f"Throughput: {result.ops_per_second:,} logs / second")
    """
    # Create logger for benchmarking
    logger = get_logger("benchmark_logging")

    # Generate test log data
    test_messages = _generate_test_log_data(log_count)

    # Measure logging performance
    start_time = time.perf_counter()
    start_memory = _get_memory_usage()

    successful_logs = 0
    failed_logs = 0

    if concurrent_threads == 1:
        # Single - threaded benchmark
        for message_data in test_messages:
            try:
                logger.info(message_data["message"], extra=message_data["extra"])
                successful_logs += 1
            except Exception:
                failed_logs += 1
    else:
        # Multi - threaded benchmark
        def log_batch(batch: list[dict[str, Any]]) -> int:
            batch_successful = 0
            for message_data in batch:
                try:
                    logger.info(message_data["message"], extra=message_data["extra"])
                    batch_successful += 1
                except Exception:
                    pass
            return batch_successful

        # Split messages into batches for threads
        batch_size = max(1, ceil(len(test_messages) / concurrent_threads))
        batches = [
            test_messages[i : i + batch_size]
            for i in range(0, len(test_messages), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            future_to_size = {
                executor.submit(log_batch, batch): len(batch)
                for batch in batches
                if batch
            }

            for future in as_completed(future_to_size):
                batch_size_hint = future_to_size[future]
                try:
                    successful_logs += future.result()
                except Exception:
                    failed_logs += batch_size_hint

    end_time = time.perf_counter()
    end_memory = _get_memory_usage()

    # Calculate metrics
    time_seconds = end_time - start_time
    memory_mb = max(0, end_memory - start_memory)
    ops_per_second = int(successful_logs / time_seconds) if time_seconds > 0 else 0
    success_rate = successful_logs / log_count if log_count > 0 else 1.0

    # Estimate throughput in MB / s (rough calculation based on average message size)
    avg_message_size_bytes = (
        sum(len(json.dumps(msg)) for msg in test_messages[:100]) / 100
    )
    throughput_mb_per_second = (ops_per_second * avg_message_size_bytes) / (1024 * 1024)

    return BenchmarkResult(
        test_name="logging_throughput",
        operations_completed=successful_logs,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
        ops_per_second=ops_per_second,
        throughput_mb_per_second=throughput_mb_per_second,
        success_rate=success_rate,
        metadata={
            "log_count": log_count,
            "concurrent_threads": concurrent_threads,
            "failed_logs": failed_logs,
            "avg_message_size_bytes": avg_message_size_bytes,
            "memory_per_log_bytes": (
                (memory_mb * 1024 * 1024) / successful_logs
                if successful_logs > 0
                else 0
            ),
        },
    )


def benchmark_json_serialization(serialization_count: int = 50000) -> BenchmarkResult:
    """
    Benchmark JSON serialization performance for log messages.

    Args:
        serialization_count: Number of JSON serializations to perform

    Returns:
        BenchmarkResult with JSON serialization metrics
    """
    # Generate complex test data for serialization
    test_data = _generate_complex_log_data(serialization_count)

    start_time = time.perf_counter()
    start_memory = _get_memory_usage()

    successful_serializations = 0
    total_bytes = 0

    for data in test_data:
        try:
            json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            total_bytes += len(json_str.encode("utf - 8"))
            successful_serializations += 1
        except Exception:
            pass

    end_time = time.perf_counter()
    end_memory = _get_memory_usage()

    # Calculate metrics
    time_seconds = end_time - start_time
    memory_mb = max(0, end_memory - start_memory)
    ops_per_second = (
        int(successful_serializations / time_seconds) if time_seconds > 0 else 0
    )
    success_rate = (
        successful_serializations / serialization_count
        if serialization_count > 0
        else 1.0
    )
    throughput_mb_per_second = (
        (total_bytes / (1024 * 1024)) / time_seconds if time_seconds > 0 else 0
    )

    return BenchmarkResult(
        test_name="json_serialization",
        operations_completed=successful_serializations,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
        ops_per_second=ops_per_second,
        throughput_mb_per_second=throughput_mb_per_second,
        success_rate=success_rate,
        metadata={
            "serialization_count": serialization_count,
            "total_bytes": total_bytes,
            "avg_bytes_per_serialization": (
                total_bytes / successful_serializations
                if successful_serializations > 0
                else 0
            ),
            "json_validity": "100%" if success_rate == 1.0 else f"{success_rate:.1%}",
        },
    )


def benchmark_io_performance(
    write_count: int = 25000, file_rotation_size: int = 10000
) -> BenchmarkResult:
    """
    Benchmark I / O throughput under high load with file rotation.

    Args:
        write_count: Number of log writes to perform
        file_rotation_size: Number of writes before rotating to new file

    Returns:
        BenchmarkResult with I / O performance metrics
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file_base = Path(temp_dir) / "benchmark"

        # Create logger with file output
        logger = get_logger("benchmark_io", log_file=f"{log_file_base}_0.log")

        # Generate test log data
        test_messages = _generate_test_log_data(write_count)

        start_time = time.perf_counter()
        start_memory = _get_memory_usage()

        successful_writes = 0
        current_file_writes = 0
        current_file_index = 0
        total_bytes_written = 0

        for message_data in test_messages:
            try:
                # Rotate log file if needed
                if current_file_writes >= file_rotation_size:
                    current_file_index += 1
                    new_log_file = f"{log_file_base}_{current_file_index}.log"

                    # Create new logger with rotated file
                    logger = get_logger(
                        f"benchmark_io_{current_file_index}", log_file=new_log_file
                    )
                    current_file_writes = 0

                # Log the message
                logger.info(message_data["message"], extra=message_data["extra"])
                successful_writes += 1
                current_file_writes += 1

                # Estimate bytes written (rough calculation)
                estimated_bytes = len(
                    json.dumps(
                        {"message": message_data["message"], **message_data["extra"]}
                    )
                )
                total_bytes_written += estimated_bytes

            except Exception:
                pass

        end_time = time.perf_counter()
        end_memory = _get_memory_usage()

        # Calculate file system metrics
        log_files = list(Path(temp_dir).glob("benchmark_*.log"))
        total_file_size = sum(f.stat().st_size for f in log_files)

        # Calculate metrics
        time_seconds = end_time - start_time
        memory_mb = max(0, end_memory - start_memory)
        ops_per_second = (
            int(successful_writes / time_seconds) if time_seconds > 0 else 0
        )
        success_rate = successful_writes / write_count if write_count > 0 else 1.0
        throughput_mb_per_second = (
            (total_file_size / (1024 * 1024)) / time_seconds if time_seconds > 0 else 0
        )

        return BenchmarkResult(
            test_name="io_performance",
            operations_completed=successful_writes,
            time_seconds=time_seconds,
            memory_mb=memory_mb,
            ops_per_second=ops_per_second,
            throughput_mb_per_second=throughput_mb_per_second,
            success_rate=success_rate,
            metadata={
                "write_count": write_count,
                "file_rotation_size": file_rotation_size,
                "files_created": len(log_files),
                "total_file_size_bytes": total_file_size,
                "avg_file_size_mb": (
                    (total_file_size / (1024 * 1024)) / len(log_files)
                    if log_files
                    else 0
                ),
                "estimated_bytes_written": total_bytes_written,
                "io_efficiency": (
                    total_file_size / total_bytes_written
                    if total_bytes_written > 0
                    else 1.0
                ),
            },
        )


def benchmark_performance_context_manager(
    operation_count: int = 5000,
) -> BenchmarkResult:
    """
    Benchmark the log_performance context manager overhead.

    Args:
        operation_count: Number of operations to time

    Returns:
        BenchmarkResult with context manager overhead metrics
    """
    logger = get_logger("benchmark_performance_cm")

    start_time = time.perf_counter()
    start_memory = _get_memory_usage()

    successful_operations = 0
    total_operation_time = 0

    for i in range(operation_count):
        try:
            with log_performance(
                f"test_operation_{i}", {"iteration": i}, logger=logger
            ) as ctx:
                # Simulate some work
                operation_start = time.perf_counter()
                time.sleep(0.001)  # 1ms simulated work
                operation_end = time.perf_counter()

                ctx["simulated_work_ms"] = (operation_end - operation_start) * 1000
                total_operation_time += operation_end - operation_start

            successful_operations += 1
        except Exception:
            pass

    end_time = time.perf_counter()
    end_memory = _get_memory_usage()

    # Calculate metrics
    time_seconds = end_time - start_time
    memory_mb = max(0, end_memory - start_memory)
    ops_per_second = (
        int(successful_operations / time_seconds) if time_seconds > 0 else 0
    )
    success_rate = (
        successful_operations / operation_count if operation_count > 0 else 1.0
    )

    # Calculate overhead (total time - actual work time)
    overhead_seconds = time_seconds - total_operation_time
    overhead_per_operation_ms = (
        (overhead_seconds * 1000) / successful_operations
        if successful_operations > 0
        else 0
    )

    return BenchmarkResult(
        test_name="performance_context_manager",
        operations_completed=successful_operations,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
        ops_per_second=ops_per_second,
        throughput_mb_per_second=0,  # Not applicable for this test
        success_rate=success_rate,
        metadata={
            "operation_count": operation_count,
            "total_operation_time_seconds": total_operation_time,
            "overhead_seconds": overhead_seconds,
            "overhead_per_operation_ms": overhead_per_operation_ms,
            "overhead_percentage": (
                (overhead_seconds / time_seconds * 100) if time_seconds > 0 else 0
            ),
        },
    )


def run_comprehensive_benchmarks() -> list[BenchmarkResult]:
    """
    Run comprehensive benchmark suite covering all performance aspects.

    Returns:
        List of BenchmarkResult objects for all tests

    Example:
        >>> results = run_comprehensive_benchmarks()
        >>> for result in results:
        ...     print(f"{result.test_name}: {result.ops_per_second:,} ops / sec")
    """
    results = []

    print("🚀 Running structured logging benchmarks...")

    # Benchmark 1: Single - threaded logging throughput
    print("📊 Running single - threaded logging throughput benchmark...")
    single_thread_result = benchmark_logging_throughput(10000, 1)
    results.append(single_thread_result)

    # Benchmark 2: Multi - threaded logging throughput
    print("🔀 Running multi - threaded logging throughput benchmark...")
    multi_thread_result = benchmark_logging_throughput(10000, 4)
    results.append(multi_thread_result)

    # Benchmark 3: JSON serialization performance
    print("🔧 Running JSON serialization benchmark...")
    json_result = benchmark_json_serialization(25000)
    results.append(json_result)

    # Benchmark 4: I / O performance with file rotation
    print("💾 Running I / O performance benchmark...")
    io_result = benchmark_io_performance(15000, 5000)
    results.append(io_result)

    # Benchmark 5: Performance context manager overhead
    print("⏱️ Running performance context manager benchmark...")
    cm_result = benchmark_performance_context_manager(2000)
    results.append(cm_result)

    return results


def _generate_test_log_data(count: int) -> list[dict[str, Any]]:
    """Generate test log data for benchmarking."""
    messages = []

    for i in range(count):
        # Vary message complexity
        if i % 4 == 0:
            # Simple message
            messages.append(
                {
                    "message": f"Simple log message {i}",
                    "extra": {"id": i, "type": "simple"},
                }
            )
        elif i % 4 == 1:
            # Medium complexity
            messages.append(
                {
                    "message": f"Processing user request {i}",
                    "extra": {
                        "user_id": i,
                        "request_type": "GET",
                        "endpoint": f"/api / users/{i}",
                        "status_code": 200,
                        "response_time_ms": 45.2 + (i % 100),
                    },
                }
            )
        elif i % 4 == 2:
            # High complexity
            messages.append(
                {
                    "message": f"Database operation completed {i}",
                    "extra": {
                        "operation": "SELECT",
                        "table": "users",
                        "rows_affected": i % 1000,
                        "query_time_ms": 123.4 + (i % 500),
                        "connection_pool": {"active": 5, "idle": 10, "total": 15},
                        "metadata": {
                            "query_hash": f"hash_{i}",
                            "cache_hit": i % 3 == 0,
                            "index_used": True,
                        },
                    },
                }
            )
        else:
            # Error message
            messages.append(
                {
                    "message": f"Error processing request {i}",
                    "extra": {
                        "error_code": "E001",
                        "error_type": "ValidationError",
                        "user_id": i,
                        "request_data": {
                            "field1": f"value_{i}",
                            "field2": i * 2,
                            "field3": [i, i + 1, i + 2],
                        },
                        "stack_trace": f"File 'test.py', line {i}, in function\n  raise ValidationError()",
                    },
                }
            )

    return messages


def _generate_complex_log_data(count: int) -> list[dict[str, Any]]:
    """Generate complex data structures for JSON serialization benchmarking."""
    data_structures = []

    for i in range(count):
        # Create increasingly complex data structures
        complexity_level = i % 5

        if complexity_level == 0:
            # Simple structure
            data_structures.append(
                {
                    "timestamp": "2025 - 01 - 01T12:00:00Z",
                    "level": "INFO",
                    "message": f"Simple message {i}",
                    "id": i,
                }
            )
        elif complexity_level == 1:
            # Nested structure
            data_structures.append(
                {
                    "timestamp": "2025 - 01 - 01T12:00:00Z",
                    "level": "INFO",
                    "message": f"Nested message {i}",
                    "user": {
                        "id": i,
                        "name": f"User {i}",
                        "email": f"user{i}@example.com",
                        "preferences": {
                            "theme": "dark",
                            "notifications": True,
                            "language": "en",
                        },
                    },
                }
            )
        elif complexity_level == 2:
            # Array structure
            data_structures.append(
                {
                    "timestamp": "2025 - 01 - 01T12:00:00Z",
                    "level": "INFO",
                    "message": f"Array message {i}",
                    "items": [
                        {"id": j, "value": f"item_{j}", "score": j * 0.1}
                        for j in range(i % 10 + 1)
                    ],
                    "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
                }
            )
        elif complexity_level == 3:
            # Deep nesting
            data_structures.append(
                {
                    "timestamp": "2025 - 01 - 01T12:00:00Z",
                    "level": "INFO",
                    "message": f"Deep nested message {i}",
                    "data": {
                        "level1": {
                            "level2": {
                                "level3": {
                                    "level4": {
                                        "value": i,
                                        "metadata": {
                                            "created": "2025 - 01 - 01",
                                            "modified": "2025 - 01 - 02",
                                            "version": "1.0",
                                        },
                                    }
                                }
                            }
                        }
                    },
                }
            )
        else:
            # Mixed complex structure
            data_structures.append(
                {
                    "timestamp": "2025 - 01 - 01T12:00:00Z",
                    "level": "ERROR",
                    "message": f"Complex error message {i}",
                    "error": {
                        "code": f"E{i:04d}",
                        "type": "ComplexError",
                        "details": {
                            "primary": f"Primary error {i}",
                            "secondary": [f"Secondary {j}" for j in range(3)],
                            "context": {
                                "request_id": f"req_{i}",
                                "session_id": f"sess_{i}",
                                "user_agent": "Mozilla / 5.0 (compatible; benchmark)",
                                "ip_address": f"192.168.1.{i % 255}",
                                "headers": {
                                    "content - type": "application / json",
                                    "authorization": "Bearer token_redacted",
                                    "x - request - id": f"req_{i}",
                                },
                            },
                        },
                    },
                    "performance": {
                        "duration_ms": 123.45 + i,
                        "memory_mb": 45.67 + (i % 100),
                        "cpu_percent": 12.34 + (i % 50),
                    },
                }
            )

    return data_structures


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if psutil is None:
        return 0.0

    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0


def generate_benchmark_report(results: list[BenchmarkResult]) -> str:
    """Generate a formatted benchmark report."""
    report = ["🔧 Structured Logging Utilities Benchmark Report", "=" * 60, ""]

    for result in results:
        report.extend(
            [
                f"📊 {result.test_name.replace('_', ' ').title()}",
                f"   Operations completed: {result.operations_completed:,}",
                f"   Time taken: {result.time_seconds:.2f}s",
                f"   Throughput: {result.ops_per_second:,} ops / sec",
                f"   Memory used: {result.memory_mb:.1f} MB",
                f"   Success rate: {result.success_rate:.1%}",
                "",
            ]
        )

        if result.throughput_mb_per_second > 0:
            report.append(
                f"   Data throughput: {result.throughput_mb_per_second:.2f} MB / s"
            )
            report.append("")

    return "\n".join(report)


def compare_benchmark_results(
    old_results: list[BenchmarkResult], new_results: list[BenchmarkResult]
) -> str:
    """Compare two sets of benchmark results."""
    report = ["🔄 Structured Logging Benchmark Comparison", "=" * 50, ""]

    # Create lookup for easy comparison
    old_lookup = {r.test_name: r for r in old_results}

    for new_result in new_results:
        test_name = new_result.test_name
        if test_name in old_lookup:
            old_result = old_lookup[test_name]

            ops_change = (
                (
                    (new_result.ops_per_second - old_result.ops_per_second)
                    / old_result.ops_per_second
                    * 100
                )
                if old_result.ops_per_second > 0
                else 0
            )
            memory_change = (
                (
                    (new_result.memory_mb - old_result.memory_mb)
                    / old_result.memory_mb
                    * 100
                )
                if old_result.memory_mb > 0
                else 0
            )

            report.extend(
                [
                    f"📈 {test_name.replace('_', ' ').title()}",
                    f"   Throughput: {ops_change:+.1f}% ({new_result.ops_per_second:,} vs {old_result.ops_per_second:,})",
                    f"   Memory: {memory_change:+.1f}% ({new_result.memory_mb:.1f} vs {old_result.memory_mb:.1f} MB)",
                    "",
                ]
            )

    return "\n".join(report)
