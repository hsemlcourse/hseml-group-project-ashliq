.PHONY: install lint format test notebook parse

install:
	pip install -r requirements.txt

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check src tests --fix
	black src tests

test:
	pytest tests

notebook:
	jupyter notebook notebooks/cp1.ipynb

parse:
	python -m src.data.parse_tmdb_movies --pages 3 --output data/processed/parsed_tmdb_movies_sample.csv
