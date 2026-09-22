# Contributing to DataConsistencyChecker

Thank you for your interest in contributing to DataConsistencyChecker! This guide will help you understand the codebase structure and how to add new features, particularly new tests.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#mainment-setup)
- [Project Structure](#project-structure)
- [Adding New Tests](#adding-new-tests)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Git for version control

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/DataConsistencyChecker.git
   cd DataConsistencyChecker
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up pre-commit hooks** (recommended)
   ```bash
   poetry run pre-commit install
   ```

4. **Verify installation**
   ```bash
   poetry run pytest -q
   ```

### Quick Start with Makefile

For convenience, a Makefile is provided with common mainment tasks:

```bash
# Install dependencies and set up pre-commit hooks
make install

# Run all checks (lint, format, type-check, tests)
make all

# Run tests
make test

# Run tests with coverage
make coverage

# Format code
make format

# Run linting
make lint

# See all available commands
make help
```

### Code Quality Tools

This project uses several tools to maintain code quality:

#### Ruff - Linting and Formatting

Ruff is an extremely fast Python linter and formatter that replaces multiple tools.

```bash
# Check for linting issues
poetry run ruff check .

# Auto-fix linting issues
poetry run ruff check . --fix

# Check formatting
poetry run ruff format --check .

# Auto-format code
poetry run ruff format .
```

#### Mypy - Type Checking

Mypy performs static type analysis to catch type-related errors.

```bash
# Run type checking
poetry run mypy .

# Run with detailed error codes
poetry run mypy . --show-error-codes
```

#### Pytest with Coverage

Run tests with coverage reporting to ensure code is properly tested.

```bash
# Run tests with coverage
poetry run pytest --cov=. --cov-report=term-missing

# Generate HTML coverage report
poetry run pytest --cov=. --cov-report=html

# View coverage report (opens in browser)
open htmlcov/index.html
```

#### Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit. Install them with:

```bash
poetry run pre-commit install
```

Run all hooks manually:

```bash
# Run on all files
poetry run pre-commit run --all-files

# Run on staged files only
poetry run pre-commit run
```

The hooks will:
- Run Ruff linter and formatter
- Run mypy type checking (on main code, not tests)
- Check for trailing whitespace
- Ensure files end with newline
- Check YAML/TOML syntax
- Detect debug statements
- Check for large files
- Run security checks with Bandit

## Project Structure

The codebase is organized into modular components for maintainability:

```
DataConsistencyChecker/
├── check_data_consistency.py     # Main DataConsistencyChecker class (~4,300 lines)
│                                  # Core orchestration, configuration, and utilities
├── test_implementations/         # Test method implementations organized by category
│   ├── base_tests_mixin.py       # Base test methods (18 methods, 1,003 lines)
│   ├── numeric_tests_mixin.py    # Numeric test methods (118 methods, 4,773 lines)
│   ├── date_tests_mixin.py       # Date test methods (36 methods, 1,043 lines)
│   ├── string_tests_mixin.py     # String test methods (118 methods, 4,187 lines)
│   ├── binary_tests_mixin.py     # Binary test methods (24 methods, 1,383 lines)
│   └── multi_column_tests_mixin.py # Multi-column test methods (18 methods, 844 lines)
├── tests_definitions/            # Test metadata organized by category
│   ├── base_tests.py             # Single/pair column tests (any type)
│   ├── numeric_tests.py          # Numeric column tests
│   ├── date_tests.py             # Date/time column tests
│   ├── string_tests.py           # String/text column tests
│   ├── binary_tests.py           # Binary column tests
│   └── multi_column_tests.py     # Multi-column & row-level tests
├── test_registry.py              # Test definition constants and registry
├── checker_utils.py              # Shared utility functions
├── display_mixin.py              # Display and output methods
├── plots_mixin.py                # Visualization methods
├── synth_data_mixin.py           # Synthetic data generation
└── tests/                        # Unit tests (one file per test)
```

### Key Components

- **Main Class** (`check_data_consistency.py`): Core orchestration, initialization, and utilities
- **Test Implementation Mixins** (`test_implementations/`): Organized test methods by category
  - Each mixin contains `_check_*` and `_generate_*` methods for its category
  - Mixins use single underscore (protected) naming to avoid Python name mangling
  - Total: 332 test methods across 6 mixin files
- **Test Definitions** (`tests_definitions/`): Metadata about each test (description, flags)
- **Display/Plotting Mixins**: Separate concerns (display, plotting, synthetic data)
- **Utilities**: Helper functions used across the codebase

### Modular Architecture Benefits

The modular structure provides several advantages:
- **Maintainability**: Each category is self-contained and easier to understand
- **Contribution**: Easier to add new tests in the appropriate category
- **Testing**: Mixins can be tested independently
- **File Size**: Main file reduced from 16,838 to 4,277 lines (74% reduction)

## Adding New Tests

Adding a new test involves three main steps:

### 1. Add Test Definition

Choose the appropriate module in `tests_definitions/` based on your test category:

- `base_tests.py` - Single columns or pairs of any type
- `numeric_tests.py` - Numeric columns
- `date_tests.py` - Date/datetime columns
- `string_tests.py` - String/text columns
- `binary_tests.py` - Binary columns
- `multi_column_tests.py` - 3+ columns or row-level tests

Add your test definition to the `get_*_tests(checker)` function:

```python
def get_numeric_tests(checker):
    return {
        # ... existing tests ...

        'YOUR_TEST_ID': (
            'Short description for progress display',
            'Full description of what the test does',
            checker._check_your_test,       # Note: single underscore
            checker._generate_your_test,     # Note: single underscore
            False,  # shortlist (include in default pattern listings)
            True,   # implemented
            True,   # fast (runs fast even with many columns)
            False   # code (assumes code/ID values)
        ),
    }
```

**Important**: Method references use single underscore (`checker._check_*`) to avoid Python name mangling issues with mixin classes.

### 2. Implement Test Logic

Add two methods to the appropriate mixin file in `test_implementations/`:

- `base_tests_mixin.py` - For single/pair column tests (any type)
- `numeric_tests_mixin.py` - For numeric column tests
- `date_tests_mixin.py` - For date/time column tests
- `string_tests_mixin.py` - For string/text column tests
- `binary_tests_mixin.py` - For binary column tests
- `multi_column_tests_mixin.py` - For multi-column tests

#### Check Method
Identifies patterns and exceptions in the data:

```python
def _check_your_test(self, test_id):  # Note: single underscore prefix
    """
    Brief description of what this test checks.

    Returns patterns_list and exceptions_list.
    """
    patterns_list = []
    exceptions_list = []

    # Loop through relevant columns
    for col_name in self.numeric_cols:
        # Your test logic here
        # Identify if pattern exists
        # Identify any exceptions to the pattern

        if pattern_found:
            patterns_list.append({
                'test_id': test_id,
                'columns': [col_name],
                # ... other metadata
            })

        if exceptions_found:
            exceptions_list.append({
                'test_id': test_id,
                'columns': [col_name],
                'rows': flagged_rows,
                # ... other metadata
            })

    return patterns_list, exceptions_list
```

#### Generator Method
Creates synthetic data to demonstrate the test:

```python
def _generate_your_test(self):  # Note: single underscore prefix
    """
    Generate synthetic data demonstrating this test.

    Creates a dataset where the pattern is present with some exceptions.
    """
    # Create example data showing the pattern
    # Include some rows that violate the pattern (exceptions)

    # Use helper method to add columns to synthetic dataframe
    self._add_synthetic_column('example_col', example_values)
```

### 3. Create Unit Test

Create a test file in `tests/test_YOUR_TEST_ID.py`:

```python
import pandas as pd
import numpy as np
from check_data_consistency import DataConsistencyChecker

def test_YOUR_TEST_ID():
    """Test YOUR_TEST_ID with synthetic data."""

    # Create test data that should trigger the pattern
    df = pd.DataFrame({
        'col1': [...],  # Data showing the pattern
        'col2': [...],  # Data with exceptions
    })

    # Run the test
    dc = DataConsistencyChecker()
    dc.init_data(df)
    dc.check_data_quality(execute_list=['YOUR_TEST_ID'])

    # Verify results
    assert len(dc.get_patterns_list()) > 0, "Should find patterns"
    assert len(dc.get_exceptions_list()) > 0, "Should find exceptions"
```

### Test Categories

Choose the right category for your test:

| Category | When to Use |
|----------|-------------|
| **Base** | Tests on single columns or pairs (any data type) |
| **Numeric** | Tests specific to numeric columns |
| **Date** | Tests specific to date/datetime columns |
| **String** | Tests specific to string/text columns |
| **Binary** | Tests on binary (two-value) columns |
| **Multi-column** | Tests on 3+ columns or row-level aggregations |

## Code Style Guidelines

### General Principles

1. **Interpretability First**: Every test should be explainable and intuitive
2. **Minimize False Positives**: Use appropriate thresholds (IQR, contamination levels)
3. **Performance Awareness**: Consider impact on datasets with many columns
4. **Consistent Naming**: Follow existing naming conventions

### Naming Conventions

- **Test IDs**: Use `UPPER_SNAKE_CASE` (e.g., `CORRELATED_NUMERIC`, `SAME_VALUES`)
- **Private Methods**: Use `__method_name` for test implementations
- **Public APIs**: Use `snake_case` for public methods
- **Variables**: Use descriptive names (`flagged_rows`, not `fr`)

### Documentation

- Add docstrings to all test implementation methods
- Explain what pattern is being detected
- Document any parameters or thresholds used
- Include examples when helpful

### Type Hints

Use Python 3.10+ type hints for new code:

```python
def check_data_quality(
    self,
    execute_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
    fast_only: bool = False
) -> None:
    """Execute data quality tests."""
    ...
```

## Testing Requirements

### Unit Tests

Every new test must include:

1. **Basic functionality test**: Verify the test finds patterns and exceptions
2. **Edge cases**: Test with empty data, single row, all nulls, etc.
3. **Different null patterns**: Test with various null value configurations

Example test structure:

```python
def test_YOUR_TEST_ID():
    """Test basic functionality."""
    # ... basic test ...

def test_YOUR_TEST_ID_edge_cases():
    """Test edge cases."""
    # Test with empty dataframe
    # Test with single row
    # Test with all nulls

def test_YOUR_TEST_ID_null_patterns():
    """Test with different null patterns."""
    # Test with random nulls
    # Test with clustered nulls
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_YOUR_TEST_ID.py

# Run with verbose output
poetry run pytest -v

# Run quickly (exit on first failure)
poetry run pytest -x
```

### Test Coverage

- Aim for high coverage of new code
- Test both the `__check_*` and `__generate_*` methods
- Verify synthetic data generation works correctly

## Pull Request Process

### Before Submitting

1. **Run all tests**: Ensure `poetry run pytest` passes
2. **Check syntax**: Ensure code compiles without errors
3. **Update documentation**: Add to README if adding significant features
4. **Add examples**: Consider adding a demo notebook if helpful

### PR Guidelines

1. **Create a descriptive title**
   - Good: "Add MONOTONIC_DIFFERENCES test for time series"
   - Bad: "New test"

2. **Provide clear description**
   ```markdown
   ## Summary
   Adds a new test that detects monotonically increasing/decreasing differences
   between consecutive values in numeric columns.

   ## Motivation
   Useful for detecting time series with constant rates of change.

   ## Testing
   - Added unit tests in tests/test_MONOTONIC_DIFFERENCES.py
   - Tested on real-world time series datasets

   ## Checklist
   - [x] Tests pass
   - [x] Documentation updated
   - [x] Code follows style guidelines
   ```

3. **Keep PRs focused**: One feature/fix per PR when possible

4. **Respond to feedback**: Be open to suggestions and iterate

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add MONOTONIC_DIFFERENCES test for time series patterns

- Detects constant rate of change in numeric columns
- Handles missing values gracefully
- Includes comprehensive unit tests

Closes #123
```

Format: `type: description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `perf`: Performance improvements

## Code Review Checklist

Before requesting review, verify:

- [ ] Code follows existing patterns and style
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (docstrings, README if needed)
- [ ] No debugging code left in (print statements, etc.)
- [ ] Type hints added for new public methods
- [ ] Performance considered (especially for tests on pairs/sets of columns)

## Future Refactoring Opportunities

The codebase is continuously evolving. Here are areas where contributions would be valuable:

### Further Modularization

The main `check_data_consistency.py` file (~16,800 lines) could be further refactored:

**Test Implementations Extraction** (High Priority)
- Move `__check_*` and `__generate_*` methods to `test_implementations/` package
- Create mixin classes by category (base, numeric, date, string, binary, multi-column)
- Main class would inherit from these mixins
- Would reduce main file to ~2,000-3,000 lines of core logic

**Benefits**:
- Easier to navigate and understand code
- Simpler to add new tests
- Better separation of concerns
- Improved code review process

**Example Structure**:
```python
# test_implementations/base_implementations.py
class BaseTestImplementations:
    def __check_missing(self, test_id):
        # Implementation here
        ...

    def __generate_missing(self):
        # Synthetic data generation
        ...
```

### Additional Type Hints

While key public APIs now have type hints, adding them throughout would help:
- Private test implementation methods
- Utility functions
- Mixin classes
- Internal helper methods

### Performance Optimizations

- Profile test execution to identify bottlenecks
- Implement more comprehensive parallel execution
- Add caching for expensive computations
- Optimize column combination generation

## Questions?

- Check existing tests for examples
- Review the [API documentation](docs/api.md)
- Look at [demo notebooks](Demo%20Notebooks/) for usage patterns
- Open an issue for clarification

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to DataConsistencyChecker! 🎉
