.PHONY: dev-env train install install-dev clean ruff check run-api freeze deploy-development deploy-staging deploy-production build test lint-fix format

IMAGE_NAME := trade-tariff-lambdas-fpo-search
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

dev-env: install-dev .git/hooks/pre-commit
	@echo
	@echo "---------------------------------------------------------"
	@echo "Development environment set up. You can activate it using"
	@echo "    source $(VENV)/bin/activate"
	@echo "---------------------------------------------------------"

train:
	${PYTHON} train.py --config search-config.toml

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

freeze: $(VENV)/bin/activate
	$(PIP) freeze --exclude hmrc_fpo_categorisation_api > requirements.txt

build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run -p 9000:8080 \
		--rm \
		--name $(IMAGE_NAME) \
		$(IMAGE_NAME) \

test-local:
	curl http://localhost:9000/2015-03-31/functions/function/invocations \
		--header 'Content-Type: application/json' \
		--data '{"path": "/fpo-code-search", "httpMethod": "POST", "body": "{\"description\": \"toothbrushes\"}"}'

shell:
	docker run \
		--rm \
		--name $(IMAGE_NAME)-shell \
		--no-healthcheck \
		--entrypoint '' \
		-it $(IMAGE_NAME) \
		/usr/bin/bash

test:
	${PYTHON} -m unittest -v -b

test-infer:
	${PYTHON} infer.py "shoes"

benchmark-goods-descriptions:
	${PYTHON} benchmark.py \
		--output json \
		--benchmark-goods-descriptions \
		--no-progress \
		--write-to-file

benchmark-classifieds:
	${PYTHON} benchmark.py \
		--output json \
		--benchmark-classifieds \
		--number-of-items 100000 \
		--no-progress \
		--write-to-file

benchmark: benchmark-goods-descriptions benchmark-classifieds

deploy-development:
	serverless deploy \
		--verbose \
		--param="custom_domain=search.dev.trade-tariff.service.gov.uk" \
		--param="certificate_domain=dev.trade-tariff.service.gov.uk" \
		--param="provisioned_concurrency=2"

deploy-staging:
	serverless deploy \
	--verbose \
	--param="custom_domain=search.staging.trade-tariff.service.gov.uk" \
	--param="certificate_domain=staging.trade-tariff.service.gov.uk" \
	--param="provisioned_concurrency=2"

deploy-production:
	serverless deploy \
	--verbose \
	--param="custom_domain=search.trade-tariff.service.gov.uk" \
	--param="certificate_domain=trade-tariff.service.gov.uk" \
	--param="provisioned_concurrency=12"
