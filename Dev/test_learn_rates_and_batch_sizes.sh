#!/bin/bash

# req. cdpt && conpt;
# requires being in the repository and in a py-env supporting the lib. dependencies (including torch)

echo "Testing learn rates 0.001, 0.01, and 0.05 for opimiser $1 and model type $2"

./run_exp_cases_different_batch_sizes.sh 0.001 "$1" "$2"
./run_exp_cases_different_batch_sizes.sh 0.01 "$1" "$2"
./run_exp_cases_different_batch_sizes.sh 0.05 "$1" "$2"
