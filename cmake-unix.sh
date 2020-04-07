#!/bin/bash

if [ -x "$(command -v cmake3)" ]; then
    cmake3 --version
    cmake3 $@
else
    cmake --version
    cmake $@
fi
make
