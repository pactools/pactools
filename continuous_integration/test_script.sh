#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python continuous_integration/print_packages_versions.py;

run_tests() {
    TEST_CMD="pytest --showlocals --pyargs"
    # Get into a temp directory to run test from the installed pactools and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR
    cd $TEST_DIR

    if [[ "$COVERAGE" == "true" ]]; then
        TEST_CMD="$TEST_CMD --with-coverage"
    fi
    $TEST_CMD sklearn

}

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
