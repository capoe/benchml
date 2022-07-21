# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PY_APP_PATH   = $(shell cd ../ && pwd)
PY_APP_NAME   = benchml
MODELS_DOC_DIR = models
MODELS_FILENAME = models.rst

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

models:
	python3 $(SOURCEDIR)/make_model_docs.py --output_dir $(SOURCEDIR)/$(MODELS_DOC_DIR)/ --model_content_name $(MODELS_FILENAME)

apidoc: models
	sphinx-apidoc -o $(SOURCEDIR)/api/ -f -e $(PY_APP_PATH)/$(PY_APP_NAME)
	rm -f $(SOURCEDIR)/api/modules.rst

html-apidoc: apidoc
	PYTHONPATH="${PYTHONPATH}:$(PY_APP_PATH)" make html

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Removing everything under '$(SOURCEDIR)/api/'..."
	@rm -Rf $(SOURCEDIR)/api/
	@echo "Removing everything under '$(SOURCEDIR)/$(MODELS_DOC_DIR)/'..."
	@rm -Rf $(SOURCEDIR)/$(MODELS_DOC_DIR)/
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
