.PHONY: dev-env train install install-dev clean ruff check run-api freeze deploy-development deploy-staging deploy-production build test lint-fix format

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

dev-env: install-dev .git/hooks/pre-commit
	@echo
	@echo "---------------------------------------------------------"
	@echo "Development environment set up. You can activate it using"
	@echo "    source $(VENV)/bin/activate"
	@echo "---------------------------------------------------------"

train: install
	${PYTHON} train.py

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

.git/hooks/pre-commit:
	$(VENV)/bin/pre-commit install

install: $(VENV)/bin/activate
	@echo ">> Installing dependencies"
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e ".[dev]"

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

lint:
	$(VENV)/bin/ruff .

lint-fix:
	$(VENV)/bin/ruff --fix .

format:
	$(VENV)/bin/ruff format .

check:
	$(VENV)/bin/ruff check .

run-api: install
	${PYTHON} api.py

freeze: $(VENV)/bin/activate
	$(PIP) freeze --exclude hmrc_fpo_categorisation_api > requirements.txt

build:
	docker build -t 382373577178.dkr.ecr.eu-west-2.amazonaws.com/tariff-fpo-search-production:latest .

test:
	${PYTHON} -m unittest -v -b

deploy-development:
	PRIVATE_ENABLED=false STAGE=development serverless deploy --verbose --param="custom_domain=search.dev.trade-tariff.service.gov.uk" --param="certificate_domain=dev.trade-tariff.service.gov.uk"

deploy-staging:
	PRIVATE_ENABLED=false STAGE=staging serverless deploy --verbose --param="custom_domain=search.sandbox.trade-tariff.service.gov.uk" --param="certificate_domain=sandbox.trade-tariff.service.gov.uk"

deploy-production:
	PRIVATE_ENABLED=true STAGE=production DOCKER_TAG=$$DOCKER_TAG serverless deploy --verbose --param="custom_domain=search.trade-tariff.service.gov.uk" --param="certificate_domain=trade-tariff.service.gov.uk"
