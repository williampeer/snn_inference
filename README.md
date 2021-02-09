# snn_inference

A repository for gradient based spiking neural network inference, as used in the ICML 2021 submission: "Parallel spiking neural network parameterinference using gradient based optimization".

## Installation

`python3 -m pip install -r requirements.txt`

## Usage
`python main.py -h`

See Models/ for models that are already implemented and their implementations.

Test/ for tests that you may want to run just to figure out how the Models can be used outside of the experiments context (or what happens under the hood).

experiments.py should give a gist about the current implementation (no interface exists as of now, but see pseudo-code in the supplementary material file).

### Sample usage
```
python main.py --experiment-type DataDriven -optim Adam -lfn frd -lr 0.05 -N 1 -ti 10
```
