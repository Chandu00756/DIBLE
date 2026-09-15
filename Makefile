PYTHON ?= python3

.PHONY: test verify package

test:
	$(PYTHON) -m unittest discover -s tests -v

verify:
	$(PYTHON) -m diblelab.cli verify --rounds 200

package:
	$(PYTHON) -m pip wheel --no-deps --wheel-dir dist .
