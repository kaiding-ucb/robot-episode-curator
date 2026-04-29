.PHONY: help install dev test test-backend test-frontend lint format clean

help:
	@echo "Targets:"
	@echo "  make install        Create venv, install Python + frontend deps"
	@echo "  make dev            Run backend (8000) + frontend (3000) dev servers"
	@echo "  make test           Run backend pytest + frontend playwright"
	@echo "  make test-backend   Run pytest only"
	@echo "  make test-frontend  Run playwright e2e only"
	@echo "  make lint           Ruff (Python) + ESLint (frontend)"
	@echo "  make format         Black + ruff --fix"
	@echo "  make clean          Remove caches and build artifacts"

install:
	bash scripts/setup.sh

dev:
	bash scripts/dev.sh

test: test-backend test-frontend

test-backend:
	. .venv/bin/activate && pytest

test-frontend:
	cd frontend && npx playwright test e2e/user-flows.spec.ts

lint:
	. .venv/bin/activate && ruff check backend tests
	cd frontend && npm run lint

format:
	. .venv/bin/activate && ruff check --fix backend tests && black backend tests

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf frontend/.next frontend/playwright-report frontend/test-results
