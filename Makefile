.PHONY: build clean deploy-development deploy-staging deploy-production test lint install-linter

build:
	cd fpo_search && make build

clean:
	cd fpo_search && make clean

deploy-development: clean build
	STAGE=development serverless deploy --verbose

deploy-staging: clean build
	STAGE=staging serverless deploy --verbose

deploy-production: clean build
	STAGE=production serverless deploy --verbose

test: clean build
	cd fpo_search && make test

check:
	cd fpo_search && make check
