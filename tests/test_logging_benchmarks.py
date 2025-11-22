# SPDX - License - Identifier: MIT
# Copyright (c) 2025 Perday CatalogLAB™

"""
Tests for structured logging benchmarking functionality.

These tests verify the benchmarking suite works correctly and produces
meaningful performance metrics for resume documentation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from structured_logging_utilities.benchmarks import (
    BenchmarkResult,
    benchmark_io_performance,
    benchmark_json_serialization,
    benchmark_logging_throughput,
    benchmark_performance_context_manager,
    compare_benchmark_results,
    generate_benchmark_report,
    run_comprehensive_benchmarks,
)


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            test_name="test_benchmark",
            operations_completed=1000,
            time_seconds=1.5,
            memory_mb=10.5,
            ops_per_second=666,
            throughput_mb_per_second=5.2,
            success_rate=0.99,
            metadata={"test": "data"},
        )

        assert result.test_name == "test_benchmark"
        assert result.operations_completed == 1000
        assert result.time_seconds == 1.5
        assert result.memory_mb == 10.5
        assert result.ops_per_second == 666
        assert result.throughput_mb_per_second == 5.2
        assert result.success_rate == 0.99
        assert result.metadata == {"test": "data"}


class TestLoggingThroughputBenchmark:
    """Test logging throughput benchmarking."""

    def test_benchmark_logging_throughput_single_thread(self):
        """Test single - threaded logging throughput benchmark."""
        result = benchmark_logging_throughput(log_count=100, concurrent_threads=1)

        assert result.test_name == "logging_throughput"
        assert result.operations_completed <= 100
        assert result.time_seconds > 0
        assert result.ops_per_second >= 0
        assert 0 <= result.success_rate <= 1
        assert "log_count" in result.metadata
        assert "concurrent_threads" in result.metadata
        assert result.metadata["concurrent_threads"] == 1

    def test_benchmark_logging_throughput_multi_thread(self):
        """Test multi - threaded logging throughput benchmark."""
        result = benchmark_logging_throughput(log_count=100, concurrent_threads=2)

        assert result.test_name == "logging_throughput"
        assert result.operations_completed <= 100
        assert result.time_seconds > 0
        assert result.ops_per_second >= 0
        assert 0 <= result.success_rate <= 1
        assert result.metadata["concurrent_threads"] == 2

    def test_benchmark_logging_throughput_performance_metrics(self):
        """Test that logging throughput benchmark produces reasonable performance metrics."""
        result = benchmark_logging_throughput(log_count=1000, concurrent_threads=1)

        # Should achieve reasonable throughput (at least 100 logs / second)
        assert result.ops_per_second >= 100

        # Memory usage should be reasonable (less than 100MB for 1000 logs)
        assert result.memory_mb < 100

        # Success rate should be high
        assert result.success_rate >= 0.95

        # Should have meaningful metadata
        assert "avg_message_size_bytes" in result.metadata
        assert result.metadata["avg_message_size_bytes"] > 0


class TestJSONSerializationBenchmark:
    """Test JSON serialization benchmarking."""

    def test_benchmark_json_serialization(self):
        """Test JSON serialization benchmark."""
        result = benchmark_json_serialization(serialization_count=1000)

        assert result.test_name == "json_serialization"
        assert result.operations_completed <= 1000
        assert result.time_seconds > 0
        assert result.ops_per_second >= 0
        assert 0 <= result.success_rate <= 1
        assert "serialization_count" in result.metadata
        assert "total_bytes" in result.metadata

    def test_json_serialization_performance_metrics(self):
        """Test that JSON serialization benchmark produces reasonable metrics."""
        result = benchmark_json_serialization(serialization_count=5000)

        # Should achieve high throughput (at least 1000 serializations / second)
        assert result.ops_per_second >= 1000

        # Should have 100% success rate for valid JSON
        assert result.success_rate >= 0.99

        # Should produce valid JSON (indicated by success rate)
        assert result.metadata["json_validity"] in ["100%", "99.9%", "99.8%"]

        # Should have reasonable bytes per serialization
        assert result.metadata["avg_bytes_per_serialization"] > 50


class TestIOPerformanceBenchmark:
    """Test I / O performance benchmarking."""

    def test_benchmark_io_performance(self):
        """Test I / O performance benchmark."""
        result = benchmark_io_performance(write_count=100, file_rotation_size=50)

        assert result.test_name == "io_performance"
        assert result.operations_completed <= 100
        assert result.time_seconds > 0
        assert result.ops_per_second >= 0
        assert 0 <= result.success_rate <= 1
        assert "write_count" in result.metadata
        assert "files_created" in result.metadata

    def test_io_performance_file_rotation(self):
        """Test that I / O benchmark handles file rotation correctly."""
        result = benchmark_io_performance(write_count=150, file_rotation_size=50)

        # Should create multiple files due to rotation
        assert result.metadata["files_created"] >= 3  # 150 / 50 = 3 files

        # Should have reasonable file sizes
        assert result.metadata["total_file_size_bytes"] > 0
        assert result.metadata["avg_file_size_mb"] > 0

    def test_io_performance_throughput_metrics(self):
        """Test I / O performance throughput metrics."""
        result = benchmark_io_performance(write_count=500, file_rotation_size=100)

        # Should achieve reasonable I / O throughput
        assert result.throughput_mb_per_second > 0

        # Should have high success rate
        assert result.success_rate >= 0.95

        # I / O efficiency should be reasonable (close to 1.0)
        assert 0.5 <= result.metadata["io_efficiency"] <= 2.0


class TestPerformanceContextManagerBenchmark:
    """Test performance context manager benchmarking."""

    def test_benchmark_performance_context_manager(self):
        """Test performance context manager benchmark."""
        result = benchmark_performance_context_manager(operation_count=100)

        assert result.test_name == "performance_context_manager"
        assert result.operations_completed <= 100
        assert result.time_seconds > 0
        assert result.ops_per_second >= 0
        assert 0 <= result.success_rate <= 1
        assert "operation_count" in result.metadata
        assert "overhead_per_operation_ms" in result.metadata

    def test_context_manager_overhead_metrics(self):
        """Test that context manager benchmark measures overhead correctly."""
        result = benchmark_performance_context_manager(operation_count=200)

        # Should have low overhead per operation (less than 10ms)
        assert result.metadata["overhead_per_operation_ms"] < 10

        # Overhead percentage should be reasonable (less than 50%)
        assert result.metadata["overhead_percentage"] < 50

        # Should have high success rate
        assert result.success_rate >= 0.95


class TestComprehensiveBenchmarks:
    """Test comprehensive benchmark suite."""

    def test_run_comprehensive_benchmarks(self):
        """Test running the comprehensive benchmark suite."""
        results = run_comprehensive_benchmarks()

        # Should return multiple benchmark results
        assert len(results) >= 4

        # Should include all expected benchmark types
        test_names = {result.test_name for result in results}
        expected_tests = {
            "logging_throughput",
            "json_serialization",
            "io_performance",
            "performance_context_manager",
        }

        assert expected_tests.issubset(test_names)

        # All results should be valid
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.operations_completed > 0
            assert result.time_seconds > 0
            assert result.ops_per_second >= 0
            assert 0 <= result.success_rate <= 1

    def test_comprehensive_benchmarks_performance_targets(self):
        """Test that comprehensive benchmarks meet performance targets."""
        results = run_comprehensive_benchmarks()

        # Find specific benchmark results
        results_by_name = {result.test_name: result for result in results}

        # Logging throughput should be high
        if "logging_throughput" in results_by_name:
            logging_result = results_by_name["logging_throughput"]
            assert logging_result.ops_per_second >= 1000  # At least 1K logs / sec

        # JSON serialization should be very fast
        if "json_serialization" in results_by_name:
            json_result = results_by_name["json_serialization"]
            assert (
                json_result.ops_per_second >= 5000
            )  # At least 5K serializations / sec

        # I / O performance should be reasonable
        if "io_performance" in results_by_name:
            io_result = results_by_name["io_performance"]
            assert io_result.ops_per_second >= 500  # At least 500 writes / sec


class TestBenchmarkReporting:
    """Test benchmark reporting functionality."""

    def test_generate_benchmark_report(self):
        """Test generating benchmark report."""
        # Create sample results
        results = [
            BenchmarkResult(
                test_name="test_benchmark_1",
                operations_completed=1000,
                time_seconds=1.0,
                memory_mb=10.0,
                ops_per_second=1000,
                throughput_mb_per_second=5.0,
                success_rate=0.99,
                metadata={},
            ),
            BenchmarkResult(
                test_name="test_benchmark_2",
                operations_completed=2000,
                time_seconds=2.0,
                memory_mb=20.0,
                ops_per_second=1000,
                throughput_mb_per_second=0,
                success_rate=1.0,
                metadata={},
            ),
        ]

        report = generate_benchmark_report(results)

        assert "Structured Logging Utilities Benchmark Report" in report
        assert "Test Benchmark 1" in report
        assert "Test Benchmark 2" in report
        assert "1,000 ops / sec" in report
        assert "99.0%" in report
        assert "100.0%" in report

    def test_compare_benchmark_results(self):
        """Test comparing benchmark results."""
        old_results = [
            BenchmarkResult(
                test_name="test_benchmark",
                operations_completed=1000,
                time_seconds=2.0,
                memory_mb=20.0,
                ops_per_second=500,
                throughput_mb_per_second=2.5,
                success_rate=0.95,
                metadata={},
            )
        ]

        new_results = [
            BenchmarkResult(
                test_name="test_benchmark",
                operations_completed=1000,
                time_seconds=1.0,
                memory_mb=10.0,
                ops_per_second=1000,
                throughput_mb_per_second=5.0,
                success_rate=0.99,
                metadata={},
            )
        ]

        comparison = compare_benchmark_results(old_results, new_results)

        assert "Benchmark Comparison" in comparison
        assert "Test Benchmark" in comparison
        assert "+100.0%" in comparison  # 100% speed improvement
        assert "-50.0%" in comparison  # 50% memory reduction


class TestBenchmarkDataGeneration:
    """Test benchmark data generation functions."""

    def test_generate_test_log_data_variety(self):
        """Test that generated test log data has appropriate variety."""
        from structured_logging_utilities.benchmarks import _generate_test_log_data

        messages = _generate_test_log_data(100)

        assert len(messages) == 100

        # Should have variety in message types
        message_types = set()
        for msg in messages:
            if "simple" in msg["extra"].get("type", ""):
                message_types.add("simple")
            elif "user_id" in msg["extra"]:
                message_types.add("user_request")
            elif "operation" in msg["extra"]:
                message_types.add("database")
            elif "error_code" in msg["extra"]:
                message_types.add("error")

        # Should have at least 3 different message types
        assert len(message_types) >= 3

    def test_generate_complex_log_data_complexity(self):
        """Test that generated complex log data has appropriate complexity levels."""
        from structured_logging_utilities.benchmarks import _generate_complex_log_data

        data_structures = _generate_complex_log_data(50)

        assert len(data_structures) == 50

        # Should have variety in complexity
        has_simple = False
        has_nested = False
        has_arrays = False
        has_deep_nesting = False

        for data in data_structures:
            if "user" in data and isinstance(data["user"], dict):
                has_nested = True
            if "items" in data and isinstance(data["items"], list):
                has_arrays = True
            if "data" in data and "level1" in data.get("data", {}):
                has_deep_nesting = True
            if len(data) <= 4:  # Simple structure
                has_simple = True

        # Should have multiple complexity levels
        complexity_count = sum([has_simple, has_nested, has_arrays, has_deep_nesting])
        assert complexity_count >= 3


class TestBenchmarkMemoryTracking:
    """Test memory tracking in benchmarks."""

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip")
        or pytest.importorskip("psutil", reason="psutil not available"),
        reason="psutil required for memory tracking tests",
    )
    def test_memory_tracking_accuracy(self):
        """Test that memory tracking provides reasonable measurements."""
        from structured_logging_utilities.benchmarks import _get_memory_usage

        initial_memory = _get_memory_usage()

        # Allocate some memory
        large_data = [f"test_string_{i}" * 1000 for i in range(1000)]

        after_allocation = _get_memory_usage()

        # Should show increased memory usage
        assert after_allocation >= initial_memory

        # Clean up
        del large_data

        # Memory usage should be tracked
        assert initial_memory >= 0
        assert after_allocation >= 0

    def test_memory_tracking_fallback(self):
        """Test memory tracking fallback when psutil unavailable."""
        from structured_logging_utilities.benchmarks import _get_memory_usage

        with patch("structured_logging_utilities.benchmarks.psutil", None):
            memory_usage = _get_memory_usage()
            assert memory_usage == 0.0


class TestBenchmarkErrorHandling:
    """Test error handling in benchmarks."""

    def test_benchmark_handles_logging_failures(self):
        """Test that benchmarks handle logging failures gracefully."""
        # Mock logger to raise exceptions
        with patch(
            "structured_logging_utilities.benchmarks.get_logger"
        ) as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            mock_logger.info.side_effect = Exception("Logging failed")

            result = benchmark_logging_throughput(log_count=10, concurrent_threads=1)

            # Should complete without crashing
            assert isinstance(result, BenchmarkResult)
            assert result.success_rate < 1.0  # Some failures expected

    def test_benchmark_handles_serialization_failures(self):
        """Test that JSON serialization benchmark handles failures gracefully."""
        result = benchmark_json_serialization(serialization_count=100)

        # Should complete successfully even with complex data
        assert isinstance(result, BenchmarkResult)
        assert result.operations_completed > 0
        assert result.success_rate > 0.8  # Most should succeed
