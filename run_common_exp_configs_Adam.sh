#!/bin/bash

# req. cdpt && conpt;
# requires being in the repository and in a py-env supporting the lib. dependencies (including torch)

echo "Running all exp. cases for learn rate: $1, $2 training iterations, the $3 optimiser, and Poisson rate: $4,
$5 model type, tau_vr: $6, loss_fn: $7"

python main.py -s SanityCheck -lr 0.05 -ti 15 -N 3 -bs 200 -tvr 12.0 -rpti 4000 -optim Adam -ipr 0.5 -mt "$1" -lfn "$2"
python main.py -s Synthetic -lr 0.05 -ti 15 -N 3 -bs 200 -tvr 12.0 -rpti 4000 -optim Adam -ipr 0.5 -mt "$1" -lfn "$2"
python main.py -s DataDriven -lr 0.05 -ti 15 -N 3 -bs 200 -tvr 12.0 -rpti 4000 -optim Adam -ipr 0.5 -mt "$1" -lfn "$2"
