#!/bin/bash

echo "Running script for all loss functions, optimiser: $1, and model $2"

./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 van_rossum_dist
./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 van_rossum_dist_per_node
./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 van_rossum_squared
./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 van_rossum_squared_per_node
./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 mse
./run_exp_cases_different_batch_sizes.sh 0.05 50 "$1" 0.5 "$2" 12.0 mse_per_node
