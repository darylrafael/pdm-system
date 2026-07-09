# ══════════════════════════════════════════════════════════════════════════
# PdM System — Makefile
# Usage: make <target>
# ══════════════════════════════════════════════════════════════════════════

.PHONY: help setup install lint format type-check security test test-unit \
        test-integration coverage run-api run-dashboard docker-up docker-down \
        docker-build clean mlflow sentinel

# ── Default ────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ""
	@echo "  PdM System — Available Commands"
	@echo "  ════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup & Install ────────────────────────────────────────────────────────
setup: ## Full first-time setup (install uv + dependencies + env file)
	@echo "→ Installing uv..."
	@pip install uv --quiet
	@echo "→ Installing project dependencies..."
	@uv sync --extra dev
	@echo "→ Setting up environment file..."
	@cp -n .env.example .env || echo "  .env already exists, skipping"
	@echo "✅ Setup complete. Edit .env before running the application."

install: ## Install/sync dependencies only
	uv sync --extra dev

# ── Code Quality ───────────────────────────────────────────────────────────
lint: ## Run ruff linter (check only)
	@echo "→ Running ruff linter..."
	uv run ruff check src/ tests/ dashboard/

lint-fix: ## Run ruff linter with auto-fix
	@echo "→ Running ruff linter with auto-fix..."
	uv run ruff check src/ tests/ dashboard/ --fix

format: ## Run ruff formatter
	@echo "→ Running ruff formatter..."
	uv run ruff format src/ tests/ dashboard/

format-check: ## Check formatting without modifying files
	uv run ruff format src/ tests/ dashboard/ --check

type-check: ## Run mypy type checker
	@echo "→ Running mypy type checker..."
	uv run mypy src/

security: ## Run bandit security scanner
	@echo "→ Running bandit security scan..."
	uv run bandit -r src/ -ll -c pyproject.toml

quality: lint format-check type-check security ## Run all quality checks
	@echo "✅ All quality checks passed."

# ── Testing ────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	@echo "→ Running all tests..."
	uv run pytest tests/

test-unit: ## Run unit tests only
	@echo "→ Running unit tests..."
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "→ Running integration tests..."
	uv run pytest tests/integration/ -v

test-fast: ## Run tests without coverage (faster)
	uv run pytest tests/ --no-cov -q

coverage: ## Generate HTML coverage report
	uv run pytest tests/ --cov=src --cov-report=html
	@echo "✅ Coverage report generated at htmlcov/index.html"

# ── CI Simulation — Run this before every PR ───────────────────────────────
ci: quality test ## Simulate full CI pipeline locally (run before pushing)
	@echo ""
	@echo "✅ CI simulation passed. Safe to push."

# ── Application ────────────────────────────────────────────────────────────
run-api: ## Start FastAPI development server
	@echo "→ Starting API server at http://localhost:8000"
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard: ## Start Streamlit dashboard
	@echo "→ Starting dashboard at http://localhost:8501"
	uv run streamlit run dashboard/app.py --server.port 8501

run-mlflow: ## Start MLflow tracking UI
	@echo "→ Starting MLflow UI at http://localhost:5000"
	uv run mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Docker ─────────────────────────────────────────────────────────────────
docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start all services via Docker Compose
	docker compose up -d
	@echo "✅ Services running:"
	@echo "   API       → http://localhost:8000"
	@echo "   Dashboard → http://localhost:8501"
	@echo "   MLflow    → http://localhost:5000"

docker-down: ## Stop all Docker Compose services
	docker compose down

docker-logs: ## Follow logs from all services
	docker compose logs -f

docker-up-dev: ## Start services with hot-reload (dev mode)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# ── SENTINEL Review Helper ─────────────────────────────────────────────────
sentinel: ## Show SENTINEL review prompt reminder
	@echo ""
	@echo "  SENTINEL CODE REVIEW"
	@echo "  ════════════════════"
	@echo "  1. Open a NEW Claude session (fresh context)"
	@echo "  2. Paste the SENTINEL system prompt"
	@echo "  3. Paste: 'Please review this PR:'"
	@echo "  4. Paste the changed files + git diff"
	@echo ""
	@echo "  git diff develop...HEAD -- src/ tests/"
	@echo ""

# ── Utilities ──────────────────────────────────────────────────────────────
clean: ## Remove all generated files and caches
	@echo "→ Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".coverage" -delete 2>/dev/null; true
	@echo "✅ Clean complete."

tree: ## Show project file structure
	@find . -not -path './.git/*' -not -path './.venv/*' \
		-not -path './mlruns/*' -not -path './__pycache__/*' \
		| sort | sed 's|[^/]*/|  |g'
