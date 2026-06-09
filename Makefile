.PHONY: install lint test cov ingest serve eval docker

install:
	pip install -e ".[dev]"

lint:
	ruff check app tests

test:
	pytest

cov:
	pytest --cov=app --cov-report=term-missing

ingest:
	ragctl ingest

serve:
	ragctl serve

eval:
	ragctl eval

docker:
	docker build -t production-rag .
