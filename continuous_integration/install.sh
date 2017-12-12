#!/bin/sh

set -e

if [[ "$RUN_FLAKE8" == "true" ]]; then
    pip install -q flake8
fi

if [[ "$SKIP_TEST" != "true" ]]; then
    conda create -n testenv --yes pip python=$PYTHON_VERSION
    source activate testenv
    conda install --yes --quiet numpy scipy matplotlib scikit-learn
    conda install --yes --quiet pytest h5py
    if [[ "$COVERAGE" == "true" ]]; then
        conda install --yes --quiet pytest-cov
    fi
    pip install -q mne
    make install
fi
