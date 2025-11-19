.PHONY: help install test lint format type-check coverage pre-commit clean all

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and pre-commit hooks
	poetry install
	poetry run pre-commit install

test:  ## Run tests
	poetry run pytest -v

test-fast:  ## Run tests quickly (exit on first failure)
	poetry run pytest -x

coverage:  ## Run tests with coverage report
	poetry run pytest --cov=. --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:  ## Run linting checks
	poetry run ruff check .

lint-fix:  ## Run linting checks and auto-fix issues
	poetry run ruff check . --fix

format:  ## Format code with ruff
	poetry run ruff format .

format-check:  ## Check code formatting without modifying files
	poetry run ruff format --check .

type-check:  ## Run type checking with mypy
	poetry run mypy . --show-error-codes

security:  ## Run security checks with bandit
	poetry run bandit -r . -c pyproject.toml

pre-commit:  ## Run all pre-commit hooks on all files
	poetry run pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks to latest versions
	poetry run pre-commit autoupdate

clean:  ## Clean cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "Cleaned cache files and build artifacts"

all: lint-fix format type-check test  ## Run all checks and tests

ci:  ## Run all CI checks locally
	@echo "Running linting..."
	@make lint
	@echo "\nRunning format check..."
	@make format-check
	@echo "\nRunning type checking..."
	@make type-check
	@echo "\nRunning tests with coverage..."
	@make coverage
	@echo "\n✅ All CI checks passed!"
