#!/bin/bash

for i in {0..5}; do
    echo options$i
    sed -i -- 's/options[0-9]/options'$i'/g' main.py
    bazel run :train
done
