#!/bin/bash

# req. cdpt && conpt;
# requires being in the repository and in a py-env supporting the lib. dependencies (including torch)

echo "Running all exp. cases for learn rate: $1, $2 training iterations, the $3 optimiser, and Poisson rate: $4,
$5 model type, tau_vr: $6, loss_fn: $7"

#python main.py -s SanityCheck -lr "$1" -ti "$2" -N 5 -bs 50 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
#python main.py -s SanityCheck -lr "$1" -ti "$2" -N 5 -bs 200 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
python main.py -s SanityCheck -lr "$1" -ti "$2" -N 5 -bs 400 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5" -lfn "$7"

#python main.py -s Synthetic -lr "$1" -ti "$2" -N 5 -bs 50 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
#python main.py -s Synthetic -lr "$1" -ti "$2" -N 5 -bs 200 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
python main.py -s Synthetic -lr "$1" -ti "$2" -N 5 -bs 400 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5" -lfn "$7"

#python main.py -s DataDriven -lr "$1" -ti "$2" -N 5 -bs 50 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
#python main.py -s DataDriven -lr "$1" -ti "$2" -N 5 -bs 200 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5"
python main.py -s DataDriven -lr "$1" -ti "$2" -N 5 -bs 400 -tvr "$6" -rpti 4000 -optim "$3" -ipr "$4" -mt "$5" -lfn "$7"
