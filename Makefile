.PHONY: test lint typecheck clean install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=creditrisk --cov-report=term-missing

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
