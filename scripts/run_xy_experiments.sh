#!/bin/bash

echo "Processing carpet_120..."
xterm -e "python src/run_xy_experiments.py data/carpet_120/trial_1.pkl"
mv xy_res.pkl xy_120_res.pkl

echo "Processing carpet_60..."
xterm -e "python src/run_xy_experiments.py data/carpet_60/trial_1.pkl"
mv xy_res.pkl xy_60_res.pkl

# echo "Processing joint..."
# xterm -e "python src/run_xy_experiments.py data/carpet_*/*.pkl"
# mv xy_res.pkl xy_joint_res.pkl