import sys

import numpy as np
import torch

from IO import save_model_params
from TargetModels import TargetEnsembleModels
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from experiments import poisson_input
from model_util import generate_model_data
from plot import plot_spike_train, plot_neuron

t = 60
# t = 15 * 60 * 1000
# for r_seed in range(4):
r_seed = 0
poisson_rate = 10.
model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=r_seed)
model_name = 'lif_ensembles_dales_{}'.format(r_seed)

print('Loaded model.')

print('Simulating data..')
model.reset_hidden_state()
gen_input = poisson_input(rate=poisson_rate, t=t, N=model.N)
gen_spiketrain = generate_model_data(model=model, inputs=gen_input)  # soft thresholded

inputs = gen_input.clone().detach()
spiketrain = gen_spiketrain.clone().detach()

plot_spike_train(spiketrain, 'Plot imported SNN', 'TEST_SPIKE_THRESH', 'export', fname='plot_'+model_name)
plot_neuron(spiketrain, 'TEST_SPIKE_THRESH', 'export', ylabel='Spike signal', fname='plot_spike_thresh_'+model_name+'.eps')
print('{} ms simulated.'.format(t))
