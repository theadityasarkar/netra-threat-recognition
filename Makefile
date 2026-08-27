# NETRA — developer convenience targets.
# SIH 2026 · PS 26189 · Ministry of Home Affairs · NCRB Women Safety Division
#
# Usage:  make <target>   (requires GNU make; on Windows use Git Bash / WSL).

.DEFAULT_GOAL := help
.PHONY: help install run test sample lint docker docker-run compose clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:         ## Install pinned dependencies
	pip install -r requirements.txt

run:             ## Launch the Streamlit app locally (http://localhost:8501)
	streamlit run app.py

test:            ## Run the full test suite
	python -m pytest -q

sample:          ## Regenerate the synthetic demo dataset (persons/links/cases/docs)
	python scripts/generate_sample_data.py --persons 220 --seed 42

lint:            ## Byte-compile every module as a fast smoke check
	python -m compileall -q app.py src

docker:          ## Build the production container image
	docker build -t netra:latest .

docker-run:      ## Run the built image on port 8501
	docker run --rm -p 8501:8501 netra:latest

compose:         ## Build and run via docker-compose (durable case store volume)
	docker compose up --build

clean:           ## Remove caches and byte-code
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
