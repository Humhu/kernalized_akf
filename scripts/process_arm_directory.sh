#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Please specify base directory"
    exit -1
fi

base_dir=$(readlink -f $1)

for f in "$base_dir"/*.bag
do
    fbase=$(basename $f .bag)
    echo "Processing $fbase..."
    python ~/Documents/akf_experiments/processing/process_arm_bag.py $f $base_dir/$fbase.pkl
done