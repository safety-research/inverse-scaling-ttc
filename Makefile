.PHONY: help install install-dev hooks lint format type-check test test-verbose test-coverage clean build run-experiment validate docs

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install the package and dependencies
	@echo "📦 Installing core dependencies..."
	pip install --user -e ./safety-tooling
	pip install --user -r requirements.txt
	@echo "✅ Core installation complete"
	@echo "💡 If you get PATH warnings, run: make setup-path"

install-dev: install ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install --user pre-commit pytest-cov pytest-xdist
	@echo "✅ Development installation complete"

install-venv: ## Install in virtual environment (recommended for clusters)
	@echo "📦 Installing in virtual environment..."
	@if [ ! -d ".venv" ]; then python -m venv .venv; fi
	@echo "🐍 Activating virtual environment..."
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e ./safety-tooling
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Virtual environment installation complete"
	@echo "💡 To activate: source .venv/bin/activate"

install-dev-venv: install-venv ## Install development dependencies in virtual environment
	@echo "📦 Installing development dependencies in venv..."
	. .venv/bin/activate && pip install pre-commit pytest-cov pytest-xdist
	@echo "✅ Development venv installation complete"

# Pre-commit hooks
hooks: ## Install pre-commit hooks
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push
	@echo "✅ Pre-commit hooks installed"

# Code quality targets
lint: ## Run all linters (ruff, pylint)
	@echo "🔍 Running ruff..."
	ruff check src/ tests/ scripts/ --fix
	@echo "🔍 Running pylint..."
	pylint src/ --disable=C0111,C0103 || true
	@echo "✅ Linting complete"

format: ## Format code with black and isort
	@echo "🎨 Formatting with black..."
	black src/ tests/ scripts/ *.py
	@echo "🎨 Sorting imports with isort..."
	isort src/ tests/ scripts/ *.py
	@echo "✅ Formatting complete"

type-check: ## Run type checking with pyright
	@echo "🔍 Running type checks..."
	pyright src/ tests/
	@echo "✅ Type checking complete"

# Testing targets
test: ## Run unit tests
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests complete"

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	python -m pytest tests/ -v -s
	@echo "✅ Verbose tests complete"

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/"
	@echo "✅ Coverage tests complete"

test-fast: ## Run tests in parallel for faster execution
	@echo "🧪 Running tests in parallel..."
	python -m pytest tests/ -n auto
	@echo "✅ Fast tests complete"

# Analysis and validation targets
validate: ## Run a quick validation with sample data
	@echo "🔍 Running validation mode..."
	python run_experiment.py experiment=main_tasks/claude4o validation.enabled=true validation.samples=5 validation.runs=1
	@echo "✅ Validation complete"

validate-vllm: ## Run a quick validation with sample data
	@echo "🔍 Running validation mode..."
	python run_experiment.py experiment=main_tasks/qwen3_8b validation.enabled=true validation.samples=5 validation.runs=1
	@echo "✅ Validation complete"

download-data: ## Download datasets from Hugging Face
	@echo "📥 Downloading datasets..."
	python scripts/download_data.py
	@echo "✅ Data download complete"

# Maintenance targets
clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	@echo "✅ Cleanup complete"

clean-results: ## Clean up experiment results (use with caution!)
	@echo "⚠️  Cleaning up results directory..."
	@read -p "Are you sure you want to delete all results? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf results/; \
		echo "✅ Results cleaned"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

# Combined targets for common workflows
check: lint type-check test ## Run all checks (lint, type-check, test)
	@echo "✅ All checks passed!"

ci: format lint type-check test ## Run CI pipeline (format, lint, type-check, test)
	@echo "🚀 CI pipeline complete!"

# Environment setup targets
setup-path: ## Add ~/.local/bin to PATH (fixes script warnings)
	@echo "🔧 Setting up PATH for local binaries..."
	@echo ""
	@echo "To fix PATH warnings, add this to your shell profile:"
	@echo "  echo 'export PATH=\"\$$HOME/.local/bin:\$$PATH\"' >> ~/.bashrc"
	@echo "  echo 'export PATH=\"\$$HOME/.local/bin:\$$PATH\"' >> ~/.zshrc"
	@echo ""
	@echo "Then reload your shell:"
	@echo "  source ~/.bashrc  # or source ~/.zshrc"
	@echo ""
	@echo "Or run this one-liner for bash:"
	@echo "  echo 'export PATH=\"\$$HOME/.local/bin:\$$PATH\"' >> ~/.bashrc && source ~/.bashrc"

check-env: ## Check environment setup
	@echo "🔍 Environment Check:"
	@echo "  Python version: $$(python --version)"
	@echo "  Python location: $$(which python)"
	@echo "  Pip version: $$(pip --version)"
	@echo "  Virtual env: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "✅ Active ($$VIRTUAL_ENV)"; else echo "❌ Not active"; fi)"
	@echo "  ~/.local/bin in PATH: $$(if echo $$PATH | grep -q ~/.local/bin; then echo "✅ Yes"; else echo "❌ No (run: make setup-path)"; fi)"
	@echo "  Available commands:"
	@echo "    ruff: $$(if command -v ruff >/dev/null 2>&1; then echo "✅ Available"; else echo "❌ Missing"; fi)"
	@echo "    black: $$(if command -v black >/dev/null 2>&1; then echo "✅ Available"; else echo "❌ Missing"; fi)"
	@echo "    pytest: $$(if command -v pytest >/dev/null 2>&1; then echo "✅ Available"; else echo "❌ Missing"; fi)"

# Development workflow targets
dev-setup: install-dev hooks ## Complete development setup
	@echo "🎯 Development environment setup complete!"
	@echo "💡 Next steps:"
	@echo "   - Copy .env.example to .env and add your API keys"
	@echo "   - Run 'make download-data' to get datasets"
	@echo "   - Run 'make validate' to test the setup"
	@echo "   - If you see PATH warnings, run 'make setup-path'"

dev-setup-venv: install-dev-venv ## Complete development setup with virtual environment (recommended for clusters)
	@echo "🎯 Virtual environment development setup complete!"
	@echo "💡 Next steps:"
	@echo "   - Activate the environment: source .venv/bin/activate"
	@echo "   - Copy .env.example to .env and add your API keys"
	@echo "   - Run 'make download-data' to get datasets"  
	@echo "   - Run 'make validate' to test the setup"

quick-check: format lint ## Quick check before commit
	@echo "⚡ Quick pre-commit check complete!"

# Environment and dependency management
update-deps: ## Update dependencies
	@echo "📦 Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip install --upgrade -e ./safety-tooling
	@echo "✅ Dependencies updated"

check-deps: ## Check for security vulnerabilities in dependencies
	@echo "🔒 Checking dependencies for security issues..."
	pip-audit || (echo "⚠️  Install pip-audit with: pip install pip-audit" && false)
	@echo "✅ Dependency check complete"

# Git workflow helpers
pre-push: check ## Run checks before pushing
	@echo "🔄 Pre-push checks complete!"

# Build targets
build: clean ## Build the package
	@echo "🔨 Building package..."
	python -m build
	@echo "✅ Build complete"

# Status and info targets
status: ## Show project status and configuration
	@echo "📊 Project Status:"
	@echo "  Python version: $$(python --version)"
	@echo "  Installed packages:"
	@pip list | grep -E "(ruff|black|isort|pylint|pyright|pytest)" || echo "    (development tools not found)"
	@echo "  Git status:"
	@git status --porcelain | head -10 || echo "    (not a git repository)"
	@echo "  Test files: $$(find tests/ -name "test_*.py" | wc -l) files"
	@echo "  Source files: $$(find src/ -name "*.py" | wc -l) files"

# Safety checks
check-secrets: ## Check for potential secrets in code
	@echo "🔍 Checking for potential secrets..."
	@git log --oneline -10 | grep -i -E "(password|secret|key|token)" || echo "✅ No obvious secrets found in recent commits"
	@grep -r -i -E "(password|secret|key|token)" --include="*.py" src/ tests/ || echo "✅ No obvious secrets found in source code"
