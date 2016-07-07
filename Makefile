# simple makefile to simplify repetetive build env management tasks

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

test:
	#$(NOSETESTS) -s -v

trailing-spaces:
	find . -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

code-analysis:
	flake8 . --ignore=E266,W503,E265

ascii:
	# find every file that contains non-ASCII characters
	# and convert these files to ASCII
	for file in `git grep -P -n -l "[\x80-\xFF]"`; \
	do \
		iconv -f utf-8 -t ascii//translit $$file -o $$file; \
	done;
