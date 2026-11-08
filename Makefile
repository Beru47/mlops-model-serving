PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

train:
	$(PY) -m mlops.pipeline

serve:
	uvicorn mlops.serve:app --port 8000

batch:
	$(PY) -m mlops.data data/in.csv --n 10000
	$(PY) -m mlops.batch --input data/in.csv --output data/out.csv

loadtest:
	$(PY) -m mlops.loadtest --n 3000 --concurrency 16

drift-traffic:
	$(PY) -m mlops.loadtest --n 1500 --shift 1.5

up:
	docker compose up --build

.PHONY: install lint test train serve batch loadtest drift-traffic up
