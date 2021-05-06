import torch

import model_util
import spike_metrics
from Models.Izhikevich import Izhikevich
from experiments import poisson_input, zip_dicts, randomise_parameters
from plot import plot_neuron, plot_spike_trains_side_by_side

static_init_parameters = {'N': 12, 'w_mean': 0.2, 'w_var': 0.3, 'a': 0.1, 'b': 0.25}
free_parameters = {'c': -62.5, 'd': 6., 'tau_g': 2.5}

snn = Izhikevich(device='cpu', parameters=zip_dicts(static_init_parameters, randomise_parameters(free_parameters)), a=0.025, b=0.25, d=8.)

inputs = poisson_input(0.5, t=500, N=static_init_parameters['N'])
membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_Izhikevich_poisson_input')

zeros = torch.zeros_like(inputs)
membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_Izhikevich_no_input')

plot_spike_trains_side_by_side(spikes, spikes_zeros, 'test_Izhikevich', title='Spiketrains random and zero input (Izhikevich)')

tau_vr = torch.tensor(20.0)
loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
