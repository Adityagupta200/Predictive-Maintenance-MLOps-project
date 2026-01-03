.PHONY: help setup pipeline train api test docs docs-serve clean

help:
	@echo "Targets:"
	@echo "  setup       Install Python deps"
	@echo "  pipeline    Run full DVC pipeline"
	@echo "  train       Run only train stage"
	@echo "  api         Run FastAPI locally"
	@echo "  test        Run tests"
	@echo "  docs        Build Sphinx docs"
	@echo "  docs-serve  Serve docs locally"
	@echo "  clean       Remove caches/build artifacts"

setup:
	pip install -r requirements.txt
	if [ -f requirements-api.txt ]; then pip install -r requirements-api.txt; fi
	pip install -r requirements-dev.txt

pipeline:
	dvc repro

train:
	dvc repro train

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docs:
	$(MAKE) -C docs html

docs-serve:
	python -m http.server -d docs/_build/html 8088

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	rm -rf docs/_build
	find . -type d -name "__pycache__" -exec rm -rf {} +
