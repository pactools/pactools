# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pactools
SOURCEDIR     = .
BUILDDIR      = ""build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


html-noplot:
	@$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."


clean:
	rm -rf auto_examples
	rm -rf modules
	rm -rf build
	rm -rf generated


install:
	rm -rf build/doctrees build/pactools.github.io
	# first clone the pactools/pactools.github.io repo because it may ask
	# for password and we don't want to delay this long build in
	# the middle of it
	# --no-checkout just fetches the root folder without content
	# --depth 1 is a speed optimization since we don't need the
	# history prior to the last commit
	git clone --no-checkout --depth 1 https://github.com/pactools/pactools.github.io.git build/pactools.github.io
	touch build/pactools.github.io/.nojekyll
	make html
	cd build/ && \
	cp -r html/* pactools.github.io && \
	cd pactools.github.io && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push
