# simple makefile to simplify repetetive build env management tasks

PYTHON ?= python
PYTEST ?= py.test --pyargs

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
	rm -rf pactools.egg-info
	rm -rf build
	rm -rf dist
	rm -rf .tags
	rm -rf .tags1
	rm -rf .coverage

############
test:
	$(PYTEST) --pyargs pactools -v

test-coverage:
	rm -rf coverage .coverage
	rm -rf dist
	rm -f `find pactools -name "*.so"`
	$(PYTHON) setup.py build_ext -i
	$(PYTEST) --pyargs --cov=pactools pactools -v --cov-config=.coveragerc

############
trailing-spaces:
	find . -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

flake8:
	flake8 . --ignore=E266,W503,E265

ascii:
	# find every file that contains non-ASCII characters
	# and convert these files to ASCII
	for file in `git grep -P -n -l "[\x80-\xFF]"`; \
	do \
		iconv -f utf-8 -t ascii//translit $$file -o $$file; \
	done;

fix: trailing-spaces ascii flake8
