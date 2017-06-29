help:
	@echo "lint - check code style with flake8"
	@echo "test - run tests quickly"
	@echo "coverage - check code coverage quickly"

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +


coverage: clean-pyc
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test:
	py.test

coverage: clean-pyc
	cp testing/matplotlibrc .
	py.test --cov=./
	rm matplotlibrc

lint:
	flake8 bonvoyage
