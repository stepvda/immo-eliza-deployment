# Immo Eliza deployment — convenience targets.
# Usage:  make <target>

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv install api streamlit test smoke docker-build docker-run clean \
        data priciness geo scrape eda retrain

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

geo:             ## Build compact geo reference (centroids + municipality polygons)
	$(PY) -m geo.build_reference

data:            ## Seed the canonical listings store from the cleaned datasets
	$(PY) -m scraper.seed

priciness:       ## (Re)build the neighbourhood-priciness surfaces (sale + rent)
	$(PY) -m geo.priciness

scrape:          ## Run a small VALIDATION crawl (see scraper/README.md to scale up)
	$(PY) -m scraper.run --sites immoweb,realo --market sale --max 50

eda:             ## EDA + spatial (grouped) cross-validation of the shipped model
	cd ml && ../$(PY) src/eda.py

retrain:         ## Full ML rebuild: preprocess -> create -> train -> tune -> evaluate
	cd ml && ../$(PY) src/preprocessing.py && ../$(PY) src/create_models.py \
	  && ../$(PY) src/train_models.py && ../$(PY) src/tune_models.py && ../$(PY) src/evaluate.py

docker-build:    ## Build the API Docker image (from the repo root context)
	docker build -f api/Dockerfile -t immo-eliza-api .

docker-run:      ## Run the API Docker image (host localhost:8010 -> container 8000)
	docker run --rm -p 8010:8000 immo-eliza-api

clean:           ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
