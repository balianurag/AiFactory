# =============================================================================
# AI Factory — Developer Makefile
#
# Works on any machine (public internet). All packages from public PyPI.
# No internal Artifactory, no corporate SSL certs required by default.
#
# Quick start (new developer):
#   make setup          ← one-time: venv, deps, .env template
#   edit .env           ← add ANTHROPIC_API_KEY or OPENAI_API_KEY
#   make db-up          ← start Qdrant + Redis in Docker
#   make dev            ← FastAPI dev server on :8000
#
# Full end-to-end Docker deployment (single command):
#   make deploy
#
# On a corporate network (Zscaler / internal Artifactory)?
#   make deploy CORPORATE=true
#   make sandbox-build CORPORATE=true
# =============================================================================

# ── Tunable variables ─────────────────────────────────────────────────────────
PYTHON      ?= python3
UV          := uv
UV_RUN      := uv run
PORT        ?= 8000          # FastAPI dev server port
SERVE_PORT  ?= 8080          # aifactory serve port
TASK        ?= "Build a FastAPI todo app with auth"
MODE        ?= cost_efficient
K           ?=               # keyword filter for test-k
F           ?= tests/        # path for test-file

# Corporate network flag.
# false (default) → public PyPI, Dockerfile.public for sandbox.
# true            → internal Artifactory index, Dockerfile (Zscaler cert).
CORPORATE   ?= false

ifeq ($(CORPORATE),true)
  SANDBOX_DOCKERFILE := docker/sandbox/Dockerfile
  PIP_INDEX_URL      := https://artifacts.expedia.biz/artifactory/api/pypi/pypi-remote/simple/
else
  SANDBOX_DOCKERFILE := docker/sandbox/Dockerfile.public
  PIP_INDEX_URL      := https://pypi.org/simple/
endif

VENV        := .venv

# ANSI colour helpers
BOLD   := \033[1m
GREEN  := \033[32m
CYAN   := \033[36m
YELLOW := \033[33m
DIM    := \033[2m
RESET  := \033[0m

# ── Default target ─────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@printf "\n$(BOLD)AI Factory — Developer Makefile$(RESET)\n"
	@printf "$(DIM)All packages from public PyPI. Pass CORPORATE=true for Zscaler network.$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; section=""} \
	      /^## / { section=substr($$0,4); printf "\n$(BOLD)%s$(RESET)\n", section } \
	      /^[a-zA-Z0-9_-]+:.*##/ { printf "  $(CYAN)%-24s$(RESET) %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)
	@printf "\n"

# =============================================================================
## Setup
# =============================================================================

.PHONY: quickstart
quickstart: ## NEW DEVELOPER: setup + databases + smoke test in one command
	@printf "\n$(BOLD)$(GREEN)AI Factory — Quick Start$(RESET)\n\n"
	$(MAKE) --no-print-directory setup
	@printf "\n$(BOLD)Step 2/3 — Starting databases...$(RESET)\n"
	$(MAKE) --no-print-directory db-up
	@printf "\n$(BOLD)Step 3/3 — Smoke test (no LLM, instant)...$(RESET)\n"
	$(UV_RUN) aifactory run "What is 2 + 2" --mode cost_efficient
	@printf "\n$(GREEN)$(BOLD)✓ Quick start complete.$(RESET)\n"
	@printf "  Databases  Qdrant:6333  Redis:6379  Postgres:5432\n"
	@printf "  Next: edit $(CYAN).env$(RESET) and add your API key, then $(CYAN)make run$(RESET)\n\n"

.PHONY: setup
setup: _check-python install env _pre-commit ## First-time setup: Python check, deps, .env, pre-commit
	@printf "\n$(GREEN)$(BOLD)✓ Setup complete.$(RESET)\n\n"
	@printf "  Next steps:\n"
	@printf "    1. $(CYAN)edit .env$(RESET)           — add your API keys\n"
	@printf "    2. $(CYAN)make db-up$(RESET)          — start Qdrant + Redis\n"
	@printf "    3. $(CYAN)make dev$(RESET)            — FastAPI dev server\n"
	@printf "    4. $(CYAN)make info$(RESET)           — verify config + keys\n"
	@printf "    5. $(CYAN)make run$(RESET)            — run the pipeline\n\n"

.PHONY: install
install: ## Sync all dependencies from public PyPI (uv sync --dev)
	@printf "$(BOLD)Installing dependencies ($(PIP_INDEX_URL))...$(RESET)\n"
	UV_INDEX_URL=$(PIP_INDEX_URL) $(UV) sync --dev
	@printf "$(GREEN)✓ Dependencies installed.$(RESET)\n"

.PHONY: env
env: ## Create .env from .env.example if it doesn't already exist
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		printf "$(YELLOW)⚠  Created .env from .env.example — fill in your API keys.$(RESET)\n"; \
	else \
		printf "$(GREEN)✓ .env already exists.$(RESET)\n"; \
	fi

.PHONY: pip-conf-check
pip-conf-check: ## Verify pip.conf and uv.toml point to public PyPI (not internal Artifactory)
	@printf "$(BOLD)Checking package index configuration...$(RESET)\n"
	@grep -q "pypi.org" pip.conf \
		&& printf "  $(GREEN)✓$(RESET) pip.conf   → public PyPI\n" \
		|| printf "  $(YELLOW)⚠$(RESET) pip.conf   → non-public index (run with CORPORATE=true?)\n"
	@grep -q "pypi.org" uv.toml \
		&& printf "  $(GREEN)✓$(RESET) uv.toml    → public PyPI\n" \
		|| printf "  $(YELLOW)⚠$(RESET) uv.toml    → non-public index\n"
	@printf "  $(DIM)Active: CORPORATE=$(CORPORATE)  PIP_INDEX=$(PIP_INDEX_URL)$(RESET)\n"

# ── Internal helpers ──────────────────────────────────────────────────────────

.PHONY: _check-python
_check-python:
	@$(PYTHON) -c "import sys; v=sys.version_info; \
		ok = v>=(3,13); \
		ok or print('\033[31mERROR: Python 3.13+ required, got {}.{}\033[0m'.format(v.major,v.minor)); \
		ok or exit(1)"
	@printf "$(GREEN)✓ Python $$($(PYTHON) --version)$(RESET)\n"

.PHONY: _pre-commit
_pre-commit:
	@$(UV_RUN) pre-commit install --install-hooks -q 2>/dev/null \
		&& printf "$(GREEN)✓ pre-commit hooks installed.$(RESET)\n" \
		|| printf "$(DIM)  pre-commit not available — skipping.$(RESET)\n"

# =============================================================================
## Dev Services  (Docker — Qdrant, Redis, Postgres)
# =============================================================================

.PHONY: db-up
db-up: ## Start Qdrant + Redis + Postgres in Docker (background)
	@printf "$(BOLD)Starting databases...$(RESET)\n"
	docker compose -f docker-compose.dbs.yml up -d
	@printf "$(GREEN)✓ Databases started.$(RESET)\n"
	@printf "  Qdrant   $(CYAN)http://localhost:6333$(RESET)\n"
	@printf "  Redis    $(CYAN)localhost:6379$(RESET)\n"
	@printf "  Postgres $(CYAN)localhost:5432$(RESET)\n"

.PHONY: db-down
db-down: ## Stop database containers (data volumes are preserved)
	docker compose -f docker-compose.dbs.yml down

.PHONY: db-restart
db-restart: db-down db-up ## Restart database containers

.PHONY: db-status
db-status: ## Show health of database containers
	@docker compose -f docker-compose.dbs.yml ps 2>/dev/null \
		|| printf "$(YELLOW)No containers running.$(RESET)\n"

.PHONY: db-wipe
db-wipe: ## Stop databases AND delete all volumes  ⚠ irreversible
	@printf "$(YELLOW)$(BOLD)⚠  This deletes all Qdrant / Redis / Postgres volumes.$(RESET)\n"
	@read -p "Type 'yes' to confirm: " ans && [ "$$ans" = "yes" ] || (printf "Aborted.\n" && exit 1)
	docker compose -f docker-compose.dbs.yml down -v
	@printf "$(GREEN)✓ Volumes deleted.$(RESET)\n"

.PHONY: db-logs
db-logs: ## Tail live logs from database containers
	docker compose -f docker-compose.dbs.yml logs -f

# =============================================================================
## Sandbox  (Docker MCP — isolated code execution)
# =============================================================================

.PHONY: sandbox-build
sandbox-build: ## Build Docker sandbox image  (CORPORATE=true for Zscaler/Artifactory)
	@printf "$(BOLD)Building sandbox image$(RESET) $(DIM)($(SANDBOX_DOCKERFILE))$(RESET)\n"
	@printf "$(DIM)Package index: $(PIP_INDEX_URL)$(RESET)\n"
	docker build \
		--file $(SANDBOX_DOCKERFILE) \
		--tag  aifactory-sandbox:latest \
		.
	@printf "$(GREEN)✓ aifactory-sandbox:latest built.$(RESET)\n"

.PHONY: sandbox-mcp-build
sandbox-mcp-build: ## Build the FastMCP server image
	docker build \
		--file docker/sandbox/Dockerfile.mcp \
		--tag  aifactory-mcp-server:latest \
		.
	@printf "$(GREEN)✓ aifactory-mcp-server:latest built.$(RESET)\n"

.PHONY: sandbox-up
sandbox-up: sandbox-build sandbox-mcp-build ## Build both sandbox images, print next step
	@printf "\n$(GREEN)✓ Sandbox images ready.$(RESET)\n"
	@printf "  Enable in .env: $(CYAN)AIFACTORY_SANDBOX_DOCKER_ENABLED=true$(RESET)\n\n"

.PHONY: sandbox-status
sandbox-status: ## Show sandbox container pool health (ready / in_use / total)
	$(UV_RUN) aifactory sandbox status

.PHONY: sandbox-test
sandbox-test: ## Smoke-test the sandbox (Python + numpy/pandas in container)
	$(UV_RUN) aifactory sandbox test

# =============================================================================
## Dev Server
# =============================================================================

.PHONY: dev
dev: ## Start FastAPI dev server with auto-reload on :8000
	$(UV_RUN) uvicorn aifactory.api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level info

.PHONY: serve
serve: ## Start persistent aifactory API server (pre-warms connections) on :8080
	$(UV_RUN) aifactory serve \
		--port $(SERVE_PORT) \
		--reload \
		--log-level debug

.PHONY: dev-full
dev-full: db-up ## Start databases then FastAPI dev server (one command)
	$(MAKE) --no-print-directory dev

# =============================================================================
## Docker End-to-End Deployment
# =============================================================================

.PHONY: deploy
deploy: pip-conf-check env deploy-build deploy-up ## Full deploy: verify → .env check → build all images → start all services
	@printf "\n$(GREEN)$(BOLD)✓ Deployment complete.$(RESET)\n"
	@printf "  Backend API  $(CYAN)http://localhost:$(PORT)/docs$(RESET)\n"
	@printf "  Health       $(CYAN)http://localhost:$(PORT)/api/health$(RESET)\n"
	@printf "  Qdrant       $(CYAN)http://localhost:6333$(RESET)\n"
	@printf "  Redis        $(CYAN)localhost:6379$(RESET)\n"
	@printf "  Postgres     $(CYAN)localhost:5432$(RESET)\n\n"
	@printf "$(DIM)Tip: make deploy-logs   — tail all container logs$(RESET)\n"
	@printf "$(DIM)Tip: make deploy-status — check container health$(RESET)\n\n"

.PHONY: deploy-build
deploy-build: ## Build backend + sandbox images  (CORPORATE=true for Zscaler)
	@printf "$(BOLD)Building backend image...$(RESET)\n"
	docker compose build backend
	@printf "$(BOLD)Building sandbox images...$(RESET)\n"
	$(MAKE) --no-print-directory sandbox-up

.PHONY: deploy-up
deploy-up: ## Start backend + all databases on the shared aifactory-net network
	@printf "$(BOLD)Starting all services...$(RESET)\n"
	@# Stop any db-only containers first to avoid network/name conflicts between
	@# docker-compose.dbs.yml (no network) and docker-compose.yml (aifactory-net)
	@docker compose -f docker-compose.dbs.yml down 2>/dev/null || true
	docker compose up -d backend qdrant redis postgres
	@printf "$(GREEN)✓ All services started (aifactory-net).$(RESET)\n"

.PHONY: deploy-down
deploy-down: ## Stop all deployed services (keep volumes)
	docker compose down
	docker compose -f docker-compose.dbs.yml down

.PHONY: deploy-logs
deploy-logs: ## Tail logs from all running services
	docker compose logs -f

.PHONY: deploy-status
deploy-status: db-status ## Show health of all running containers
	@docker compose ps 2>/dev/null || true

.PHONY: deploy-restart
deploy-restart: deploy-down deploy-up ## Restart all services without rebuilding

.PHONY: deploy-full-stack
deploy-full-stack: ## Start frontend + backend + databases (requires AiFactory-ui repo)
	docker compose up --build

# =============================================================================
## Pipeline
# =============================================================================

.PHONY: run
run: ## Run the agent pipeline  (e.g. make run TASK="Build a REST API" MODE=balanced)
	$(UV_RUN) aifactory run "$(TASK)" --mode $(MODE)

.PHONY: run-best
run-best: ## Run with best (highest quality) model selection
	$(UV_RUN) aifactory run "$(TASK)" --mode best

.PHONY: run-fast
run-fast: ## Run cost_efficient mode — fast + cheap
	$(UV_RUN) aifactory run "$(TASK)" --mode cost_efficient

.PHONY: ask
ask: ## Quick single-agent answer  (e.g. make ask TASK="What is LangGraph?")
	$(UV_RUN) aifactory ask "$(TASK)"

.PHONY: status
status: ## List recent pipeline runs
	$(UV_RUN) aifactory status

# =============================================================================
## Testing
# =============================================================================

.PHONY: test
test: ## Run the full test suite
	$(UV_RUN) pytest tests/ -q

.PHONY: test-v
test-v: ## Run tests with verbose output
	$(UV_RUN) pytest tests/ -v

.PHONY: test-fast
test-fast: ## Run tests, stop on first failure  (-x)
	$(UV_RUN) pytest tests/ -q -x

.PHONY: test-k
test-k: ## Filter tests by keyword  (e.g. make test-k K="complexity_gate")
	$(UV_RUN) pytest tests/ -q -k "$(K)"

.PHONY: test-file
test-file: ## Run a specific file  (e.g. make test-file F=tests/test_dynamic_pipeline.py)
	$(UV_RUN) pytest "$(F)" -v

.PHONY: test-cov
test-cov: ## Run tests with HTML coverage report → htmlcov/index.html
	$(UV_RUN) pytest tests/ \
		--cov=aifactory \
		--cov-report=term-missing \
		--cov-report=html:htmlcov
	@printf "$(GREEN)✓ Coverage report: $(CYAN)htmlcov/index.html$(RESET)\n"

# =============================================================================
## Code Quality
# =============================================================================

.PHONY: lint
lint: ## Check code with ruff (no writes)
	$(UV_RUN) ruff check src/ tests/

.PHONY: lint-fix
lint-fix: ## Check and auto-fix ruff violations
	$(UV_RUN) ruff check src/ tests/ --fix

.PHONY: format
format: ## Format code with ruff
	$(UV_RUN) ruff format src/ tests/

.PHONY: format-check
format-check: ## Check formatting without writing (CI mode)
	$(UV_RUN) ruff format src/ tests/ --check

.PHONY: typecheck
typecheck: ## Run mypy static type checking over src/
	$(UV_RUN) mypy src/

.PHONY: check
check: lint format-check typecheck ## Run all quality checks (lint + format + types)
	@printf "$(GREEN)✓ All quality checks passed.$(RESET)\n"

.PHONY: fix
fix: lint-fix format ## Auto-fix lint violations and reformat

# =============================================================================
## AI Factory CLI
# =============================================================================

.PHONY: info
info: ## Show AI Factory config + API key status
	$(UV_RUN) aifactory info

.PHONY: memory-health
memory-health: ## Check memory system: Qdrant, embedding provider, Redis, collections
	$(UV_RUN) aifactory memory health

.PHONY: memory-stats
memory-stats: ## Show memory layer entry counts (agent / team / global)
	$(UV_RUN) aifactory memory stats

.PHONY: memory-search
memory-search: ## Search memory  (e.g. make memory-search Q="FastAPI auth")
	$(UV_RUN) aifactory memory search "$(Q)"

.PHONY: memory-clear
memory-clear: ## Clear agent memory layer (team + global preserved)
	$(UV_RUN) aifactory memory clear --layer agent --yes

.PHONY: agents
agents: ## List all agent roles, tiers, and default models
	$(UV_RUN) aifactory agents list

.PHONY: models
models: ## List all models and per-role assignments
	$(UV_RUN) aifactory models list

.PHONY: rl-stats
rl-stats: ## Show RL-lite adaptive model performance scores
	$(UV_RUN) aifactory rl stats

.PHONY: rl-reset
rl-reset: ## Reset all RL performance records (fresh start)
	$(UV_RUN) aifactory rl reset --yes

# =============================================================================
## Cleanup
# =============================================================================

.PHONY: clean-cache
clean-cache: ## Remove __pycache__, .pytest_cache, .mypy_cache (keep venv)
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null; true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache
	@printf "$(GREEN)✓ Caches cleared.$(RESET)\n"

.PHONY: clean
clean: clean-cache ## Remove venv, build artefacts, and caches (re-run setup after this)
	rm -rf $(VENV) dist build *.egg-info htmlcov .coverage
	@printf "$(GREEN)✓ Cleaned. Run $(CYAN)make setup$(RESET)$(GREEN) to rebuild.$(RESET)\n"

.PHONY: clean-dumps
clean-dumps: ## Delete all local pipeline run dumps  ⚠ irreversible
	@printf "$(YELLOW)⚠  This deletes all run dumps in dumps/*.json$(RESET)\n"
	@read -p "Type 'yes' to confirm: " ans && [ "$$ans" = "yes" ] || (printf "Aborted.\n" && exit 1)
	@find dumps/ -name "*.json" -delete 2>/dev/null; true
	@printf "$(GREEN)✓ Dumps cleared.$(RESET)\n"

# =============================================================================
## CI
# =============================================================================

.PHONY: ci
ci: install check test ## Full CI pipeline: install + lint + typecheck + tests
	@printf "$(GREEN)$(BOLD)✓ CI passed.$(RESET)\n"

.PHONY: certs
certs: ## (Expedia only) Regenerate Zscaler CA bundle from macOS keychain
	@mkdir -p certs
	@security find-certificate -a -p /Library/Keychains/System.keychain > certs/expedia-ca-bundle.pem
	@cat /opt/homebrew/etc/openssl@3/cert.pem certs/expedia-ca-bundle.pem > certs/combined-ca-bundle.pem
	@printf "$(GREEN)✓ certs/combined-ca-bundle.pem regenerated$(RESET)\n"
	@printf "$(DIM)Not needed outside the Expedia corporate network.$(RESET)\n"
