.PHONY: lint lint-fix

lint: ## Run ruff linter and formatter
	uv run ruff check src/ tests/ scripts/ api/ services/
	uv run ruff format --check src/ tests/ scripts/ api/ services/

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/ scripts/ api/ services/
	uv run ruff format src/ tests/ scripts/ api/ services/
