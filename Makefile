.PHONY: clean clean-test clean-pyc clean-build docs help build-docker-test-image
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint:
	pre-commit run --all-files --show-diff-on-failure

build-docker-test-image: requirements_dev.txt
	# rerun this when pip requirements change

	# if this is slow, check size of local directory
	# may have accidentally installed a bunch of python environments in .tox/
	# to fix: make clean-test

	docker build -t genetools-test .

## run tests locally using the docker image that matches Github Actions platform
# tests/results/tmp will be used for basetemp - can't use tests/results because then mpl results will be cleared out.
test: build-docker-test-image
	mkdir -p tests/results/tmp
	docker run --rm -it -v $$(pwd):/src genetools-test pytest --cov=./ --cov-report term --cov-report xml --run-snapshots --mpl --mpl-results-path=tests/results --basetemp=tests/results/tmp -vv;

# run tests locally, without docker, therefore omitting the snapshot tests
test-without-figures:
	# note: snapshot tests not run!
	pytest --cov=./ --cov-report term --cov-report xml -vv;

## regenerate baseline figures
regen-snapshot-figures: build-docker-test-image
	docker run --rm -it -v $$(pwd):/src genetools-test pytest --mpl-generate-path=tests/baseline --run-snapshots --snapshot-update;

## regenerate saved test data (and baseline figures)
regen-test-data: build-docker-test-image
	docker run --rm -it -v $$(pwd):/src genetools-test pytest --mpl-generate-path=tests/baseline --run-snapshots --snapshot-update --regenerate-anndata;

coverage: ## check code coverage quickly with the default Python
	coverage run --source genetools -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/genetools.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ genetools
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
