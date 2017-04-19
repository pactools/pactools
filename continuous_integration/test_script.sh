#!/bin/sh

set -e

if [[ "$SKIP_TEST" != "true" ]]; then
    if [[ "$COVERAGE" == "true" ]]; then
        python run_pytest.py --cov=pactools
    else
        python run_pytest.py
    fi

fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    make flake8
fi
