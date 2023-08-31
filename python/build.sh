#!/bin/bash

BASEDIR=$(dirname "$0")

# Create Python virtual environment if it doesn't exist
if ! [ -d $BASEDIR/../.env ]; then
    python -m venv $BASEDIR/../.env
fi

# Activate python virtual environment
source "$BASEDIR/../.env/bin/activate"

pip install -r $BASEDIR/../requirements.txt

# Build python library if it doesn't exist
maturin build --release --features python

pip install $BASEDIR/../target/wheels/onnx_rust*