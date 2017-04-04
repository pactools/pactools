# simple makefile to simplify repetetive build env management tasks

PYTHON ?= python

in: inplace

inplace:
	$(PYTHON) setup.py build_ext -i

install:
	$(PYTHON) setup.py install

develop:
	$(PYTHON) setup.py develop

clean:
	rm -f `find pactools -name "*.so"`
	rm -f `find pactools -name "*.pyc"`
	rm -rf `find pactools -name "*__pycache__*"`
	rm -rf build dist
	rm -rf doc/auto_examples doc/modules doc/build
	rm -rf .tags .tags1 .coverage

############
test:
	make ascii
	$(PYTHON) run_pytest.py
	make flake8

test-coverage:
	rm -rf coverage .coverage
	rm -rf dist
	rm -f `find pactools -name "*.so"`
	$(PYTHON) setup.py build_ext -i
	$(PYTHON) run_pytest.py --cov=pactools
	make flake8

############
trailing-spaces:
	find . -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

flake8:
	flake8 . --ignore=E266,W503,E265,E123 --exclude=setup.py

ascii:
	# find every file that contains non-ASCII characters
	# and convert these files to ASCII
	for file in `git grep -P -n -l "[\x80-\xFF]"`; \
	do \
		iconv -f utf-8 -t ascii//translit $$file -o $$file; \
	done;

fix: trailing-spaces ascii flake8
