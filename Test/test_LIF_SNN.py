import torch

import model_util
import spike_metrics
from Models.LIF import LIF
from experiments import poisson_input, zip_dicts, randomise_parameters
from plot import plot_neuron, plot_spiketrains_side_by_side

static_parameters = {'N': 3}
free_parameters = {'w_mean': 0.2, 'w_var': 0.3, 'tau_m': 1.5, 'tau_g': 4.0, 'v_rest': -60.0}

snn = LIF(device='cpu', parameters=zip_dicts(static_parameters, free_parameters))

inputs = poisson_input(0.5, t=500, N=static_parameters['N'])
# inputs = torch.cat((torch.tensor([12*[4.]]), torch.zeros((100, static_parameters['N']))))
# inputs = 4. * torch.ones((500, static_parameters['N']))
membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
plot_neuron(membrane_potentials.data, title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_LIF_poisson_input')

zeros = torch.zeros_like(inputs)
membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
plot_neuron(membrane_potentials_zeros.data, title='LIF neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_LIF_no_input')

plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input')

tau_vr = torch.tensor(2.0)
loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
