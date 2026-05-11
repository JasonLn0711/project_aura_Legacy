PYTHON ?= python
PYTHONPATH ?= src

.PHONY: check test compile build clean

check: compile test

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -W error::ResourceWarning -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall src tests

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
