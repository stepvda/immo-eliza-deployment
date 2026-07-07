# Immo Eliza deployment — convenience targets.
# Usage:  make <target>

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv install api streamlit test smoke docker-build docker-run clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:            ## Create the virtual environment
	python3.13 -m venv $(VENV)

install: venv    ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

api:             ## Run the FastAPI backend (http://localhost:8010, docs at /docs)
	cd api && ../$(VENV)/bin/uvicorn app:app --reload --port 8010

streamlit:       ## Run the Streamlit frontend (http://localhost:8501)
	$(VENV)/bin/streamlit run streamlit/app.py

test:            ## Run the test suite
	$(VENV)/bin/pytest -q

smoke:           ## Smoke-test a running API (default: localhost:8010)
	$(PY) scripts/smoke_test_api.py $(URL)

docker-build:    ## Build the API Docker image
	docker build -t immo-eliza-api ./api

docker-run:      ## Run the API Docker image (host localhost:8010 -> container 8000)
	docker run --rm -p 8010:8000 immo-eliza-api

clean:           ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
