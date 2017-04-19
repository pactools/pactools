#!/bin/sh

set -e

if [[ "$COVERAGE" == "true" ]]; then
    bash <(curl -s https://codecov.io/bash)
fi
