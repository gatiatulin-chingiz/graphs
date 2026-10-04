.PHONY: install notebook pipeline-sql pipeline-excel docker-build docker-run

PYTHON ?= python
INPUT_DIR ?= .
OUTPUT_DIR ?= ./output

install:
	$(PYTHON) -m pip install -r requirements.txt

notebook:
	$(PYTHON) -m jupyter notebook notebook.ipynb

pipeline-sql:
	$(PYTHON) main.py pipeline sql --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

pipeline-excel:
	$(PYTHON) main.py pipeline excel --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

pipeline-excel-viz:
	$(PYTHON) main.py pipeline excel --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR) --group $(GROUP)

docker-build:
	docker build -t graph-analytics .

docker-run:
	docker run --rm --env-file .env \
		-v "$(CURDIR)/$(INPUT_DIR):/data/input" \
		-v "$(CURDIR)/$(OUTPUT_DIR):/data/output" \
		graph-analytics pipeline excel
