.PHONY: dev-env train install install-dev clean ruff check run-api freeze deploy-development deploy-staging deploy-production

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
	${PYTHON} -m venv $(VENV)

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

lint: install-dev
	$(VENV)/bin/ruff .

check: install-dev
	$(VENV)/bin/ruff check .

run-api: install
	${PYTHON} api.py

freeze: $(VENV)/bin/activate
	$(PIP) freeze --exclude hmrc_fpo_categorisation_api > requirements.txt

build:
	docker build -t 382373577178.dkr.ecr.eu-west-2.amazonaws.com/tariff-fpo-search-production:latest .

login:
	aws ecr get-login-password --region eu-west-2 |  docker login --username AWS --password-stdin  (aws ssm get-parameter --name "/development/FPO_SEARCH_ECR_URL" --with-decryption --region eu-west-2 | jq -r '.Parameter.Value')

deploy-development:
	STAGE=development serverless deploy --verbose

deploy-staging:
	STAGE=staging serverless deploy --verbose

deploy-production:
	STAGE=production serverless deploy --verbose
