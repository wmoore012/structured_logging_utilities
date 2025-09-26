<!-- SPDX-License-Identifier: MIT
Copyright (c) 2024 Perday Labs -->

# Contributing to Structured Logging Utilities

Thank you for your interest in contributing to Structured Logging Utilities! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/perdaycatalog/structured-logging-utilities.git
   cd structured-logging-utilities
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,benchmark]"
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **ruff**: Fast linting and formatting
- **black**: Code formatting
- **mypy**: Type checking

Run all checks:
```bash
# Linting
ruff check src tests

# Formatting
ruff format src tests

# Type checking
mypy src/structured_logging_utilities
```

### Testing

We maintain high test coverage (>90%) with comprehensive test suites:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=structured_logging_utilities --cov-report=html

# Run specific test categories
pytest tests/test_core.py -v
pytest tests/test_benchmarks.py -v
pytest tests/test_integration.py -v
```

### Benchmarking

Performance is critical for logging utilities. Run benchmarks to verify performance:

```bash
python -c "
from structured_logging_utilities.benchmarks import run_comprehensive_benchmarks, generate_benchmark_report
results = run_comprehensive_benchmarks()
print(generate_benchmark_report(results))
"
```

## Contribution Guidelines

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Ensure all tests pass** and maintain coverage
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Code Requirements

- **Type hints**: All public functions must have type hints
- **Docstrings**: All public functions must have docstrings with examples
- **Tests**: New features require comprehensive tests
- **Performance**: Logging utilities must maintain high performance
- **Backward compatibility**: Avoid breaking changes

### Performance Standards

New features must meet these performance targets:

- **Logging throughput**: >5,000 log entries/second
- **JSON serialization**: >10,000 serializations/second
- **Memory usage**: <100MB for 24-hour retention
- **Context manager overhead**: <5ms per operation

### Documentation Standards

- **README examples**: Must work copy-paste
- **API documentation**: Include usage examples
- **Performance metrics**: Document benchmark results
- **Error handling**: Document exception scenarios

## Issue Reporting

When reporting issues, please include:

- **Python version** and operating system
- **Package version** of structured-logging-utilities
- **Minimal reproduction case**
- **Expected vs actual behavior**
- **Performance impact** (if applicable)

## Feature Requests

For new features, please provide:

- **Use case description**: Why is this needed?
- **API proposal**: How should it work?
- **Performance considerations**: Impact on existing performance
- **Backward compatibility**: Will it break existing code?

## Security

For security issues, please email security@perdaycatalog.com instead of creating public issues.

## Performance Optimization

When contributing performance improvements:

1. **Benchmark before and after** using the built-in benchmarking suite
2. **Document the optimization technique** used
3. **Verify no regression** in other performance metrics
4. **Update benchmark targets** if significantly improved

Example benchmark comparison:
```python
from structured_logging_utilities.benchmarks import compare_benchmark_results

# Run benchmarks before and after changes
old_results = run_comprehensive_benchmarks()
# ... make changes ...
new_results = run_comprehensive_benchmarks()

print(compare_benchmark_results(old_results, new_results))
```

## Code Review Checklist

Before submitting, verify:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No linting errors (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Performance benchmarks meet targets
- [ ] Documentation is updated
- [ ] Examples work as shown
- [ ] Backward compatibility maintained

## Release Process

Releases are automated through GitHub Actions:

1. **Version bump**: Update version in `pyproject.toml`
2. **Create tag**: `git tag v0.2.0 && git push origin v0.2.0`
3. **Automated release**: CI builds and publishes to PyPI

## Questions?

- **General questions**: Create a GitHub issue
- **Development help**: Check existing issues and discussions
- **Security concerns**: Email security@perdaycatalog.com

Thank you for contributing to making structured logging better for everyone!
