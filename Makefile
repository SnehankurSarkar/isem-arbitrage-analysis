.PHONY: install test sample reproduce

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

sample:
	python scripts/00_make_synthetic_sample.py --rows 2500 --output data/sample/isem_synthetic_sample.csv

reproduce:
	python scripts/01_run_reproduction_pipeline.py --data data/private/MarketData_2022-2026.parquet --outputs outputs --figures reports/figures
