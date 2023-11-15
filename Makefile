VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Set up the developer environment
.PHONY: dev-env
dev-env: install-dev .git/hooks/pre-commit
	@echo
	@echo "---------------------------------------------------------"
	@echo "Development environment set up. You can activate it using"
	@echo "    source $(VENV)/bin/activate"
	@echo "---------------------------------------------------------"

# Run the training
.PHONY: train
train: install
	./venv/bin/python3 train.py

## Create the venv
$(VENV)/bin/activate:
	python3 -m venv $(VENV)

## Pre-commit hooks
.git/hooks/pre-commit:
	$(VENV)/bin/pre-commit install

## Install dependencies for production
.PHONY: install
install: $(VENV)/bin/activate
	@echo ">> Installing dependencies"
	$(PIP) install --upgrade pip
	$(PIP) install -e .

## Install dependencies for development
.PHONY: install-dev
install-dev: install
	$(PIP) install -e ".[dev]"

## Delete all temporary files
clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf **/.pytest_cache
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf $(VENV)

## Lint using ruff
.PHONY: ruff
ruff: install-dev
	$(VENV)/bin/ruff .

## Run checks (ruff + test)
.PHONY: check
check: install-dev
	$(VENV)/bin/ruff check .

## Run the API locally
.PHONY: run-api
run-api: install
	${PYTHON} api.py

## Freeze the requirements to requirements.txt
.PHONY: freeze
freeze: $(VENV)/bin/activate
	$(PIP) freeze --exclude hmrc_fpo_categorisation_api > requirements.txt